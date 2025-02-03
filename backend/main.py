import asyncio
import logging
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

origins = [
    "https://promptpal-ten.vercel.app/" # If using a custom domain
]

# Add CORS middleware (for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Automatically select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and tokenizer
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # Or use "gpt2" for smaller model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    logger.info(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Model failed to load. Check your model name and internet connection.")

# Define input request model
class ChatRequest(BaseModel):
    input_text: str

# Function to generate response
def generate_response(input_text: str):
    try:
        logger.info(f"Generating response for input: {input_text}")

        # Use structured prompt to guide the model
        prompt = f"Question: {input_text}\nAnswer:"
        
        with torch.inference_mode():
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                inputs,
                max_length=80,  # Increase length slightly
                min_length=40,  # Ensure it completes at least 40 tokens
                pad_token_id=tokenizer.eos_token_id,  # Ensure a proper stopping point
                do_sample=True,
                top_k=50,  
                top_p=0.9,  
                temperature=0.7,  # More coherent responses
                repetition_penalty=1.2,  # Reduce repeated words
                num_return_sequences=1,  # Generate only one response
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ensure response makes sense
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()

        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during response generation: {str(e)}")
        return "Sorry, I couldn't process your request."

# Define chat API endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        input_text = request.input_text.strip()
        if not input_text:
            return {"error": "Input text cannot be empty."}

        logger.info(f"Received input: {input_text}")
        response = await asyncio.to_thread(generate_response, input_text)
        
        return {"response": response}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": "An error occurred while processing your request."}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
