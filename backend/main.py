from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Adjust origins in production for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

@app.post("/chat")
async def chat(input_text: str):
    try:
        inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
        outputs = model.generate(inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
