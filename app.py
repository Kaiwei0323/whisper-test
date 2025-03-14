from flask import Flask, render_template, request, jsonify
import pyaudio
import speech_recognition as sr
import threading
from whisper.transcribe import cli
import logging
import sys

app = Flask(__name__)

# Global variables to control the listening loop
is_listening = False
recognized_text = ""

# Set logging level to suppress unnecessary logs but retain important logs
logging.basicConfig(level=logging.INFO)  # Log at INFO level
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)  # Suppress werkzeug logs except errors

# Function to continuously listen to the microphone
def listen_to_microphone(mic_index):
    global recognized_text, is_listening
    while is_listening:
        print("Listening...")
        recognized_text = cli(mic_index)

# Fetch the list of microphones
def get_microphone_list():
    return [(index, name) for index, name in enumerate(sr.Microphone.list_microphone_names())]

# Route to start/stop listening and return live transcription
@app.route('/', methods=['GET', 'POST'])
def index():
    global is_listening, recognized_text
    mic_list = get_microphone_list()  # Get list of available microphones
    if request.method == 'POST':
        try:
            mic_index = int(request.form['mic_index'])
            if not is_listening:
                # Start listening
                is_listening = True
                threading.Thread(target=listen_to_microphone, args=(mic_index,), daemon=True).start()
                return render_template('index.html', recognized_text=recognized_text, listening=True, mic_list=mic_list)
            else:
                # Stop listening
                is_listening = False
                return render_template('index.html', recognized_text=recognized_text, listening=False, mic_list=mic_list)
        
        except ValueError:
            return render_template('index.html', recognized_text="Invalid microphone index", listening=False, mic_list=mic_list)
    
    return render_template('index.html', recognized_text=recognized_text, listening=False, mic_list=mic_list)

# Route to fetch the latest transcription
@app.route('/get_transcription')
def get_transcription():
    global recognized_text
    return jsonify({"transcription": recognized_text})

if __name__ == "__main__":
    # Default values
    host = "0.0.0.0"
    port = 5003
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if "--host=" in arg:
                host = arg.split("=")[1]
            if "--port=" in arg:
                port = int(arg.split("=")[1])

    app.run(host=host, port=port, debug=True)

