from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import os
import torch
from flask_cors import CORS
#pip install flask-cors

app = Flask(__name__)
CORS(app, origins=None) # Explicitly allow all origins

# Model loading - Handle potential errors more gracefully
current_directory = os.path.dirname(os.path.abspath(__file__))  # Verify this path!
#filename = "stories260K.gguf"  # Replace with your actual filename
filename = "Qwen2.5-0.5B-Instruct-Q8_0.gguf"  # Replace with your actual filename
model = None  # Initialize model outside the try block
tokenizer = None  # Initialize tokenizer outside the try block

try:
    tokenizer = AutoTokenizer.from_pretrained(current_directory, gguf_file=filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(current_directory, gguf_file=filename).to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Model loading failed: {e}")  # More descriptive message
    # Instead of exit, set a flag or handle the error in generate_text()
    model_loaded = False  # Flag to indicate model load status
else:
    model_loaded = True

@app.errorhandler(Exception)  # General exception handler for the entire app
def handle_exception(e):
    print(f"General Exception: {e}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/generate', methods=['POST'])
def generate_text():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded.  Check server logs.'}), 500

    try:
        data = request.get_json()
        prompt = data.get('prompt')
        prompts = data.get('prompts')  # For batch requests

        if prompt:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input to device
            generation_output = model.generate(
                **inputs,
                max_new_tokens=data.get('max_new_tokens', 50),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.95),
                do_sample=data.get('do_sample', True),
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            return jsonify({'generated_text': generated_text})

        elif prompts:
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)  # Move input to device
            generation_output = model.generate(
                **inputs,
                max_new_tokens=data.get('max_new_tokens', 50),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.95),
                do_sample=data.get('do_sample', True),
                pad_token_id=tokenizer.eos_token_id
            )
            generated_texts = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
            return jsonify({'generated_texts': generated_texts})

        else:
            return jsonify({'error': 'Either "prompt" or "prompts" must be provided'}), 400

    except Exception as e:
        print(f"Error during generation: {e}")  # Print error for debugging
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])  # Root route for documentation
def documentation():
    return jsonify({
        "message": "Welcome to the Qwen2.5-0.5B-Instruct-Q8_0 API!",
        "usage": {
            "/generate": {
                "method": "POST",
                "description": "Generates text from a prompt or a batch of prompts.",
                "request_body": {
                    "prompt": "string (optional). The input prompt.",
                    "prompts": "list of strings (optional). A list of input prompts for batch generation.",
                    "max_new_tokens": "integer (optional, default 50). The maximum number of tokens to generate.",
                    "temperature": "float (optional, default 0.7). Controls the randomness of the generation. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more deterministic.",
                    "top_p": "float (optional, default 0.95). Nucleus sampling probability threshold.  A value of 1.0 means no nucleus sampling.",
                    "do_sample": "boolean (optional, default True). Whether to use sampling (True) or greedy decoding (False)."
                },
                "response": {
                    "generated_text": "string (if 'prompt' was provided). The generated text.",
                    "generated_texts": "list of strings (if 'prompts' was provided). The list of generated texts.",
                    "error": "string (if an error occurred). An error message."
                },
                "example_single": {
                    "request": 'curl -X POST -H "Content-Type: application/json" -d \'{"prompt": "Translate \'Hello, how are you?\' to French."}\' http://127.0.0.1:5000/generate',
                    "response": '{"generated_text": "Bonjour, comment allez-vous?"}'  # Example, actual output may vary
                },
                "example_batch": {
                    "request": 'curl -X POST -H "Content-Type: application/json" -d \'{"prompts": ["Translate \'Hello, how are you?\' to French.", "What is the capital of France?"]}\' http://127.0.0.1:5000/generate',
                    "response": '{"generated_texts": ["Bonjour, comment allez-vous?", "Paris"]}'  # Example, actual output may vary
                }
            }
        }
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')  # Set debug to False in production
