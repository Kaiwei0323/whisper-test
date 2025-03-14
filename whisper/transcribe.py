import io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['USE_TF'] = '-1'
os.environ['USE_TORCH'] = '-1'
os.environ['USE_JAX'] = '-1'
import sys
sys.path.append(os.getcwd())
import argparse
import warnings
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
import tqdm
import pyaudio
import soundfile as sf
import speech_recognition as sr

from whisper.model import load_model, available_models
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt

if TYPE_CHECKING:
    from whisper.model import Whisper

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'
    

def print_info(start, end, text):
    """
    Returns the transcribed text along with start and end timestamps as a string.
    """
    return f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"


def transcribe(
    *,
    model: "Whisper",
    audio: Union[str, np.ndarray],
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper
    """

    mel: np.ndarray = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        segment = pad_or_trim(mel, N_FRAMES)
        _, probs = model.detect_language(segment)
        decode_options["language"] = max(probs, key=probs.get)
        if verbose is not None:
            print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    mel = mel[np.newaxis, ...]
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: np.ndarray) -> List[DecodingResult]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results = model.decode(segment, options)

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                options = DecodingOptions(**kwargs, temperature=t)
                retries = model.decode(segment[needs_fallback], options)
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]

        return results

    seek = 0
    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)
    time_precision = input_stride * HOP_LENGTH / SAMPLE_RATE
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
        *, start: float, end: float, text_tokens: np.ndarray, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        # Return the formatted text with timestamps
        return print_info(start, end, text)

    num_frames = mel.shape[-1]
    previous_seek_value = seek

    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result = decode_with_fallback(segment)[0]
            tokens = result.tokens

            if no_speech_threshold is not None:
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    should_skip = False
                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: np.ndarray = np.greater_equal(tokens, tokenizer.timestamp_begin)
            consecutive = np.add(np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0], 1)
            if len(consecutive) > 0:
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0] - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1] - tokenizer.timestamp_begin
                    )
                    segment_output = add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    if segment_output:
                        print(segment_output)  # Directly print the output when available

                    last_slice = current_slice
                last_timestamp_position = (
                    tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(list(tokens[: last_slice + 1]))
            else:
                duration = segment_duration
                tokens = np.asarray(tokens) if isinstance(tokens, list) else tokens
                timestamps = tokens[
                    np.ravel_multi_index(np.nonzero(timestamp_tokens), timestamp_tokens.shape)
                ]
                if len(timestamps) > 0:
                    last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                segment_output = add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )
                if segment_output:
                    print(segment_output)  # Directly print the output when available

                seek += segment.shape[-1]
                all_tokens.extend(list(tokens))

            if not condition_on_previous_text or result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)

            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)


def cli(device_index):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Arguments setup
    parser.add_argument("--mode", type=str, default="mic", choices=["audio", "mic"], help="Audio file(audio) or Microphone(mic)")
    parser.add_argument("--audio", nargs="*", type=str, help="Specify the path to audio files (mp4, mp3, etc.)")
    parser.add_argument("--model", default="tiny", choices=available_models(), help="Whisper model")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print progress and debug messages")
    parser.add_argument("--language", type=str, default="en", choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    
    
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple lengt normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    # Other arguments
    args = parser.parse_args().__dict__
    model_name = args.pop("model")
    output_dir = args.pop("output_dir")
    mode = args.pop("mode")
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_name)

    if mode == 'audio':
        audios = args.pop("audio")
        if not audios:
            print(f"{Color.RED}ERROR:{Color.RESET} Specify audio files.")
            sys.exit(0)
        for audio_path in audios:
            result = transcribe(model=model, audio=audio_path, **args)
            audio_basename = os.path.basename(audio_path)

            # Save output files
            with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
                write_txt(result["segments"], file=txt)
            with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
                write_vtt(result["segments"], file=vtt)
            with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
                write_srt(result["segments"], file=srt)

    elif mode == 'mic':
        args.pop("audio")
        p = pyaudio.PyAudio()
        # device_index = int(input("Please input your microphone Device ID: "))
        recognizer = sr.Recognizer()
        mic = sr.Microphone(sample_rate=16_000, device_index=device_index)
        try:
            print("Speak now! (CTRL + C to exit the application)")
            with mic as audio_source:
                recognizer.adjust_for_ambient_noise(audio_source)
                audio = recognizer.listen(audio_source)
            try:
                wav_data = audio.get_wav_data()
                wav_stream = io.BytesIO(wav_data)
                audio_array, _ = sf.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                result = transcribe(model=model, audio=audio_array, **args)
                return result['text']
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                pass
        except KeyboardInterrupt:
            pass
