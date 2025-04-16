import sys
import subprocess
import os
import io
import logging
import traceback
import json
import uuid
import base64
import asyncio
import random
import time
from PIL import Image
from typing import Dict, List, Any, Set, Optional

# Install dependencies
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi import Request
    import nest_asyncio
    from pyngrok import ngrok
    import uvicorn
    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaNextForConditionalGeneration
except ImportError:
    print("Installing required packages...")
    packages = [
        "fastapi", "uvicorn", "pyngrok", "pillow", "nest-asyncio", "requests", 
        "torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Reimport after installation
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi import Request
    import nest_asyncio
    from pyngrok import ngrok
    import uvicorn
    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaNextForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
NGROK_AUTH_TOKEN = "2vUt1PacxXv6GFFWl1xsRqeXILR_6JQeqUkXJt5tzJZK68R4p"  # Replace with your token
PORT = 8000

# Model configuration
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b"
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Create context store to maintain conversation history
context_store: Dict[str, List[Dict[str, str]]] = {}

# Create mapping between client_ids and session_ids
client_session_mapping: Dict[str, str] = {}

# Store for visual context descriptions
visual_context_store: Dict[str, str] = {}

# Store for tracking active visual context updates
visual_context_updating: Dict[str, bool] = {}

# ===== LOAD AI MODELS =====
def load_models():
    logger.info("Loading AI models...")
    
    print("Loading LLAVA model for image captioning...")
    # Load LLAVA-Next model for image captioning
    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading LLM model for text generation...")
    # Load Llama-3 model for text generation
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    print("Models loaded successfully!")
    return llava_processor, llava_model, llm_tokenizer, llm_model

# Load models globally (will be available to all functions)
llava_processor, llava_model, llm_tokenizer, llm_model = load_models()

# ===== FASTAPI APP =====
app = FastAPI(title="Image Analysis WebSocket API",
              description="API with dual WebSocket channels for frames and queries using local AI models",
              version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== WEBSOCKET CONNECTION MANAGERS =====
class FrameConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Frame client {client_id} connected. Total frame connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Frame client {client_id} disconnected. Remaining frame connections: {len(self.active_connections)}")

class QueryConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Query client {client_id} connected. Total query connections: {len(self.active_connections)}")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Query client {client_id} disconnected. Remaining query connections: {len(self.active_connections)}")

# Initialize managers
frame_manager = FrameConnectionManager()
query_manager = QueryConnectionManager()

# ===== API ENDPOINTS =====
@app.get("/")
async def home():
    return {"message": "Dual WebSocket API Ready (Frames + Queries)", "status": "success"}

# ===== CORE FUNCTIONS =====
async def generate_caption_with_llava(image_path: str, client_id: str, frame_index: int) -> str:
    """Generate caption using LLAVA-NEXT for a given image"""
    try:
        logger.info(f"Generating caption for frame {frame_index} using LLAVA")
        
        # Load image
        image = Image.open(image_path)
        
        # Prepare prompt
        prompt = "Describe what you see in this image in one detailed sentence."
        
        # Process image and text
        inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        
        # Generate caption
        with torch.no_grad():
            outputs = llava_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        
        # Process output
        caption = llava_processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's reply without the prompt
        if prompt in caption:
            caption = caption.split(prompt)[1].strip()
        
        print(f"frame {frame_index} received from /frames/{client_id}")
        print(f"caption: {caption}")
        
        return caption
        
    except Exception as e:
        logger.exception(f"Error generating caption with LLAVA: {str(e)}")
        return f"Error generating caption: {str(e)}"

async def generate_response_with_llama(prompt: str, session_id: str) -> str:
    """Generate response using LLAMA-3 model"""
    try:
        logger.info(f"Generating response with LLM for session {session_id}")
        
        # Format prompt for Llama-3
        formatted_prompt = f"<|system|>\nYou are a helpful visual assistant that analyzes images and provides insights and next steps.\n<|user|>\n{prompt}\n<|assistant|>"
        
        # Tokenize input
        inputs = llm_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # Generate text
        with torch.no_grad():
            outputs = llm_model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        # Decode response
        full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = full_response.split("<|assistant|>")[-1].strip()
        
        print(f"\nResponse from LLM: {response}")
        print(f"Response sent to /query/{session_id}")
        
        return response
        
    except Exception as e:
        logger.exception(f"Error generating response with LLM: {str(e)}")
        return f"Error generating response: {str(e)}"

def create_enhanced_prompt(base_question: str, session_id: str) -> str:
    """
    Creates an enhanced prompt that includes:
    1. Context from previous interactions
    2. The current question
    3. Request for next steps
    4. Visual context (if available)
    """
    # Get context for this session
    context = context_store.get(session_id, [])

    # Get visual context if available
    visual_context = visual_context_store.get(session_id, "")
    visual_context_section = ""
    if visual_context:
        visual_context_section = f"\nVisual context: {visual_context}\n"

    # Build context string from previous interactions
    context_string = ""
    if len(context) > 1:  # Check if there's actual history (more than just the current query)
        # Include all previous interactions except the current one
        # We'll use the most recent 5 interactions to avoid exceeding context limits
        recent_context = context[-6:-1] if len(context) > 6 else context[:-1]

        # Format context as a conversation transcript
        context_string = "Previous conversation history:\n"

        for i, item in enumerate(recent_context):
            user_query = item.get("user_query", "")
            ai_response = item.get("ai_response", "")

            if user_query:
                context_string += f"User: {user_query}\n"
            if ai_response:
                # Include a concise version of AI responses (first paragraph)
                first_paragraph = ai_response.split("\n\n")[0] if "\n\n" in ai_response else ai_response[:150]
                context_string += f"Assistant: {first_paragraph}\n"

        context_string += "\n"  # Add spacing between context and current query

    # Construct the enhanced prompt with context, visual context, and request for next steps
    if base_question and base_question != "Analyze this image:":
        enhanced_prompt = f"""{context_string}{visual_context_section}Current query: {base_question}

After analyzing, provide what is the exact next step to do? The next actionable step in 1 or 2 or 3 sentence only.
Remember to consider all the previous conversation history when making your recommendation."""
    else:
        enhanced_prompt = f"""{context_string}{visual_context_section}Please analyze this image and provide the following:
 Any key observations or points of interest. (1 sentence only)
 The next actionable step in 1 or 2 or 3 sentence only.
 Only need to provide the next steps. You can ask any questions to get more context.
 Consider all previous conversation history when making your recommendation.
"""

    return enhanced_prompt

def log_middleware_prompt(text_prompt, session_id):
    """
    Logs the structure of the prompt being sent to LLAMA-3 with a simplified structure
    """
    # Get the full conversation history for logging
    context = context_store.get(session_id, [])

    # Format previous queries for logging
    previous_queries = []
    if len(context) > 1:  # Make sure there's actual history
        for item in context[:-1]:  # Exclude current query
            if "user_query" in item:
                previous_queries.append(item["user_query"])

    # Current query is the last item in context
    current_query = context[-1]["user_query"] if context else "No query"

    # Build conversation history for logging
    conversation_history = {
        "previous_queries": previous_queries,
        "current_query": current_query,
        "total_turns": len(context),
        "conversation_flow": []
    }

    # Add detailed conversation flow (last 5 turns)
    recent_context = context[-6:-1] if len(context) > 6 else context[:-1]
    for item in recent_context:
        conversation_history["conversation_flow"].append({
            "user": item.get("user_query", ""),
            "ai": item.get("ai_response", "")[:50] + "..." if item.get("ai_response", "") else ""
        })

    # Add visual context (if available)
    visual_context = visual_context_store.get(session_id, "")
    print("\n" + "=" * 30)

    # Print the simplified structure as requested
    print("\nMIDDLEWARE PROMPT STRUCTURE")
    print(json.dumps(conversation_history, indent=2))
    print(f"\nvisual_context: {visual_context}")
    print("\nAfter analyzing, provide what is the exact next step to do? The response should be shorter and concise, considering all the previous conversation history while making response.")
    
    print("\n" + "=" * 30)

    # Also log a formatted version for debugging purposes (won't be printed to console)
    logger.info(f"\n{'='*50}\nMIDDLEWARE PROMPT STRUCTURE:\n{json.dumps(conversation_history, indent=2)}\n{'='*50}")
    
    return conversation_history

# ===== WEBSOCKET ENDPOINTS =====
@app.websocket("/frames/{client_id}")
async def frames_websocket_endpoint(websocket: WebSocket, client_id: str):
    """Endpoint for receiving frame streams from client"""
    await frame_manager.connect(websocket, client_id)
    
    try:
        # Get or create session ID
        if client_id in client_session_mapping:
            session_id = client_session_mapping[client_id]
        else:
            session_id = str(uuid.uuid4())
            client_session_mapping[client_id] = session_id
            
        # Initialize visual context updating flag
        visual_context_updating[client_id] = True
        
        # Initialize context stores if needed
        if session_id not in context_store:
            context_store[session_id] = []
        
        if session_id not in visual_context_store:
            visual_context_store[session_id] = ""
        
        frame_count = 0
        
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            image_data = data.get("image_data", "")
            
            if not image_data:
                continue
                
            # Check if visual context updating is paused (user query received)
            if not visual_context_updating.get(client_id, True):
                logger.info(f"Visual context updating stopped for {client_id}, ignoring frame")
                continue
                
            # Process the frame
            try:
                frame_count += 1
                
                # Decode image from base64
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                    
                image_bytes = base64.b64decode(image_data)
                
                # Create temporary directory if it doesn't exist
                temp_dir = "./temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save image to temporary file
                temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                with open(temp_path, "wb") as f:
                    f.write(image_bytes)
                
                # Generate caption with LLAVA model
                caption = await generate_caption_with_llava(temp_path, client_id, frame_count)
                
                # Update visual context with new caption
                if visual_context_store[session_id]:
                    visual_context_store[session_id] += " " + caption
                else:
                    visual_context_store[session_id] = caption
                
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {str(cleanup_error)}")
                    
            except Exception as e:
                logger.exception(f"Error processing frame {frame_count} for client {client_id}")
                
    except WebSocketDisconnect:
        frame_manager.disconnect(client_id)
        logger.info(f"Frame WebSocket for client {client_id} disconnected")
    except Exception as e:
        logger.exception(f"Frame WebSocket error for client {client_id}: {str(e)}")
        frame_manager.disconnect(client_id)

@app.websocket("/query/{client_id}")
async def query_websocket_endpoint(websocket: WebSocket, client_id: str):
    """Endpoint for handling query/response interactions"""
    await query_manager.connect(websocket, client_id)
    
    try:
        # Get or create session ID
        if client_id in client_session_mapping:
            session_id = client_session_mapping[client_id]
        else:
            session_id = str(uuid.uuid4())
            client_session_mapping[client_id] = session_id
            
        # Initialize context stores if needed
        if session_id not in context_store:
            context_store[session_id] = []
            
        while True:
            # Receive query data
            data = await websocket.receive_json()
            message_type = data.get("type", "")
            
            # Handle connection initialization
            if message_type == "connect":
                # Use provided session_id or existing one
                provided_session_id = data.get("session_id")
                if provided_session_id:
                    session_id = provided_session_id
                    client_session_mapping[client_id] = session_id
                
                await query_manager.send_message(client_id, {
                    "type": "connection_established",
                    "session_id": session_id,
                    "message": "Connected to image analysis service"
                })
                
            # Handle query processing
            elif message_type == "query":
                # First, stop visual context updating
                visual_context_updating[client_id] = False
                print(f"\nReceived the User Query: {data.get('question', '')}")
                print("visual_context Updation Stopped.")
                
                question = data.get("question", "Analyze this image:")
                
                # Add current query to context
                context_store[session_id].append({
                    "user_query": question,
                    "timestamp": str(uuid.uuid4())
                })
                
                # Check for reset context command
                if question.lower().strip() in ["reset context", "clear context"]:
                    context_store[session_id] = [{"user_query": "Context reset requested", "timestamp": str(uuid.uuid4())}]
                    visual_context_store[session_id] = ""  # Clear visual context too
                    await query_manager.send_message(client_id, {
                        "type": "analysis_result",
                        "status": "success",
                        "analysis": "Context has been reset. Your conversation history has been cleared.",
                        "session_id": session_id
                    })
                    continue
                
                # Notify client that processing has started
                await query_manager.send_message(client_id, {
                    "type": "processing_status",
                    "status": "processing",
                    "message": "Processing your query..."
                })
                
                try:
                    # Print full visual context to console
                    print(f"\nvisual_context:{visual_context_store[session_id]}\n")
                    
                    # Create enhanced prompt
                    enhanced_question = create_enhanced_prompt(question, session_id)
                    
                    # Log the middleware prompt structure
                    log_middleware_prompt(enhanced_question, session_id)
                    
                    # Generate response from LLM
                    processed_response = await generate_response_with_llama(enhanced_question, session_id)
                    
                    # Update the latest context entry with the response
                    if context_store[session_id]:
                        context_store[session_id][-1]["ai_response"] = processed_response
                    
                    # Keep context size manageable (keep only last 10 interactions)
                    if len(context_store[session_id]) > 10:
                        context_store[session_id] = context_store[session_id][-10:]
                    
                    # Send response to client
                    await query_manager.send_message(client_id, {
                        "type": "analysis_result",
                        "status": "success",
                        "analysis": processed_response,
                        "session_id": session_id
                    })
                    
                except Exception as e:
                    logger.exception(f"Error processing query for client {client_id}")
                    await query_manager.send_message(client_id, {
                        "type": "analysis_result",
                        "status": "error",
                        "error": str(e),
                        "session_id": session_id
                    })
                    
                # Re-enable visual context updating after query is processed
                # Only if specifically requested in the data
                if data.get("resume_visual_context", False):
                    visual_context_updating[client_id] = True
                    
            # Handle ping messages
            elif message_type == "ping":
                await query_manager.send_message(client_id, {
                    "type": "pong",
                    "timestamp": str(uuid.uuid4()),
                    "session_id": session_id
                })
                
    except WebSocketDisconnect:
        query_manager.disconnect(client_id)
        logger.info(f"Query WebSocket for client {client_id} disconnected, but session mapping preserved")
    except Exception as e:
        logger.exception(f"Query WebSocket error for client {client_id}: {str(e)}")
        query_manager.disconnect(client_id)

# ===== START SERVER =====
def start_server():
    # Configure ngrok
    try:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(PORT).public_url
        logger.info(f"🌐 Public URL: {public_url}")
        print(f"\n{'='*50}\n🌐 SERVER RUNNING AT: {public_url}\n{'='*50}\n")
        print("\nWebSocket URLs:")
        print(f"Frames: wss://{public_url.split('://')[1]}/frames/{{client_id}}")
        print(f"Query: wss://{public_url.split('://')[1]}/query/{{client_id}}")
    except Exception as e:
        logger.error(f"Ngrok error: {str(e)}")
        print(f"Ngrok error: {str(e)}")
        print(f"Server will still run locally:")
        print(f"Frames: ws://127.0.0.1:{PORT}/frames/{{client_id}}")
        print(f"Query: ws://127.0.0.1:{PORT}/query/{{client_id}}")

    # Error handler for unexpected exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "An unexpected error occurred"}
        )

    # Apply nest_asyncio for Jupyter compatibility (if needed)
    nest_asyncio.apply()

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    start_server()