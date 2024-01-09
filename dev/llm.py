from flask import Flask, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)   
torch.set_default_device("cuda")

# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

model = None
tokenizer = None

def load_model():
    global model
    global tokenizer
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    print("Model loaded")

@app.route('/generate', methods=['POST'])
def generate():
    global model
    global tokenizer
    if model is None or tokenizer is None:
        load_model()
    prompt = request.form.get('prompt')
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}  # Ensure inputs are on the same device as the model
    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    return text

if __name__ == '__main__':
    load_model()
    app.run(port=5000, debug=True)