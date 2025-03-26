import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Define paths for the original models and the quantized models
decoder_model_path = 'tiny_decoder_11.onnx'
encoder_model_path = 'tiny_encoder_11.onnx'

# Quantize the models and save the output to new files
quantize_dynamic(decoder_model_path, 'quantized_tiny_decoder_11.onnx', weight_type=QuantType.QUInt8)
quantize_dynamic(encoder_model_path, 'quantized_tiny_encoder_11.onnx', weight_type=QuantType.QUInt8)

print("Quantization completed and saved the models.")

