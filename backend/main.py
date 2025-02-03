import os
import logging
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS settings
origins = [
    "https://promptpal-ten.vercel.app",  # Add your Vercel frontend domain here
]

# Add CORS middleware (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Only allow your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use CPU to avoid GPU memory issues
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Load a smaller model (distilgpt2)
MODEL_NAME = "distilgpt2"  # Smaller, more efficient model
tokenizer, model = None, None

# Load model function
def load_model():
    global tokenizer, model
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        # Set pad_token to eos_token
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Successfully loaded model: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Model failed to load. Check your model name and internet connection.")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Define input request model
class ChatRequest(BaseModel):
    input_text: str

# Function to generate response with optimized memory usage
def generate_response(input_text: str):
    try:
        logger.info(f"Generating response for input: {input_text}")

        # Use structured prompt to guide the model
        prompt = f"Question: {input_text}\nAnswer:"

        # Tokenize input and return tensors with attention_mask
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=50, padding=True).to(device)

        # Pass the attention_mask and pad_token_id to the generation function
        outputs = model.generate(
            inputs,
            attention_mask=inputs.new_ones(inputs.shape),  # Ensure attention mask is set correctly
            pad_token_id=tokenizer.eos_token_id,           # Set pad_token_id to eos_token_id for open-ended generation
            max_length=50,                                 # Further reduced max length for efficiency
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()

        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during response generation: {str(e)}")
        return "Sorry, I couldn't process your request."

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Chat endpoint to process user input
@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received input: {request.input_text}")
    response = generate_response(request.input_text)
    return {"response": response}

# Main entry point for running the app
if __name__ == "__main__":
    # Retrieve PORT from environment, default to 8000 for local testing
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)  # Use 1 worker for Render deployment
