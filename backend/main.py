import asyncio
import logging
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins in production for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer and model on the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Function to handle model inference
def generate_response(input_text: str):
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/chat")
async def chat(input_text: str):
    try:
        logger.info(f"Received input: {input_text}")
        # Offload model inference to a separate thread
        response = await asyncio.to_thread(generate_response, input_text)
        logger.info(f"Response generated: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

# Health check endpoint for monitoring
@app.get("/health")
def health():
    return {"status": "ok"}
