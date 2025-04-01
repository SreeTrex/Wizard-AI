import torch
import uuid
import base64
import io
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from flask import Flask, request
from flask_socketio import SocketIO, emit
from pyngrok import ngrok
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Constants
MAX_WORKERS = 4  # Adjust based on your CPU cores
CAPTION_BATCH_SIZE = 10  # Number of captions to batch before combining

# Load models
blip_processor = BlipProcessor.from_pretrained("/content/blip_tokenizer.json")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "/content/blip_pytorch_model.bin", 
    config="/content/blip_config.json"
)

llama_tokenizer = AutoTokenizer.from_pretrained("/content/llama_tokenizer.json")
llama_model = AutoModelForCausalLM.from_pretrained(
    "/content/Ilama3_pytorch_model.bin",
    config="/content/llama_config.json",
    torch_dtype=torch.float16
)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Session storage
sessions = {}

# Frame processing queue and worker thread
frame_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
processing_active = True

def generate_session_id():
    return str(uuid.uuid4())

def process_single_frame(frame):
    """Process a single frame and generate a caption"""
    inputs = blip_processor(images=frame, return_tensors="pt")
    with torch.no_grad():  # Disable gradient calculation for inference
        caption = blip_model.generate(**inputs, max_length=50)
    caption_text = blip_processor.decode(caption[0], skip_special_tokens=True)
    return caption_text

def frame_processing_worker():
    """Worker thread to process frames from the queue"""
    while processing_active:
        try:
            task = frame_queue.get(timeout=1)
            if task is None:  # Sentinel value to exit
                break
                
            session_id, image, metadata = task
            
            # Process the frame
            caption = process_single_frame(image)
            
            # Store the result
            with threading.Lock():
                if session_id in sessions:
                    sessions[session_id]["captions"].append({
                        "caption": caption,
                        "timestamp": metadata.get("timestamp"),
                        "is_first": metadata.get("first_frame", False),
                        "is_last": metadata.get("last_frame", False)
                    })
                    
                    # If we have enough captions, consolidate them
                    if len(sessions[session_id]["captions"]) >= CAPTION_BATCH_SIZE:
                        consolidate_captions(session_id)
                        
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            if task:
                frame_queue.task_done()

def consolidate_captions(session_id):
    """Consolidate multiple individual captions into a collective description"""
    with threading.Lock():
        if session_id not in sessions:
            return
            
        captions_to_process = sessions[session_id]["captions"][-CAPTION_BATCH_SIZE:]
        
        # Extract timestamps for context
        start_time = captions_to_process[0]["timestamp"]
        end_time = captions_to_process[-1]["timestamp"]
        
        # Combine individual captions
        caption_texts = [c["caption"] for c in captions_to_process]
        
        # Create a prompt for LLaMa to summarize the captions
        prompt = f"""Summarize the following sequence of video frame descriptions into a cohesive, collective description.
        Focus on the main objects, actions, and changes in the scene.
        
        Frame descriptions:
        {' | '.join(caption_texts)}
        
        Collective description:"""
        
        # Generate consolidated description
        inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = llama_model.generate(**inputs, max_new_tokens=100)
        collective_description = llama_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Save to consolidated captions
        sessions[session_id]["consolidated_captions"].append({
            "description": collective_description,
            "start_time": start_time,
            "end_time": end_time,
            "frame_count": len(captions_to_process)
        })
        
        # Emit the consolidated caption to the client
        socketio.emit("caption_update", {
            "description": collective_description,
            "timestamp_range": f"{start_time}-{end_time}"
        }, room=session_id)

@socketio.on("connect")
def handle_connect():
    session_id = request.sid  
    sessions[session_id] = {
        "title": None,
        "history": [],
        "captions": [],
        "consolidated_captions": []
    }
    emit("session_id", {"session_id": session_id})
    print(f"Client connected: {session_id}")

@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id in sessions:
        del sessions[session_id]
        print(f"Client disconnected: {session_id}")

@socketio.on("frames")
def handle_frame(data):
    session_id = request.sid  
    if session_id not in sessions:
        return

    # Decode frame
    try:
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Add frame to processing queue
        metadata = data.get("meta", {})
        frame_queue.put((session_id, image, metadata))
        
        # Acknowledge receipt
        emit("frame_received", {"status": "queued", "queue_size": frame_queue.qsize()})
        
    except Exception as e:
        print(f"Error handling frame: {str(e)}")
        emit("error", {"message": f"Error processing frame: {str(e)}"})

@socketio.on("query_response")
def handle_query(data):
    session_id = request.sid
    if session_id not in sessions:
        return

    query = data.get("query")
    
    # Initialize title if first query
    if sessions[session_id]["title"] is None:
        sessions[session_id]["title"] = query  

    # Build context from consolidated captions
    with threading.Lock():
        caption_context = "\n".join(
            f"Scene {i+1} ({cap['start_time']}-{cap['end_time']}): {cap['description']}" 
            for i, cap in enumerate(sessions[session_id]["consolidated_captions"][-3:])
        )

    prompt = f"""Conversation Context:
    Title: {sessions[session_id]['title']}
    Recent Visual Scenes:
    {caption_context}
    Previous Dialogue:
    {" ".join(sessions[session_id]["history"][-4:])}
    User Query: {query}"""

    # Generate response
    inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = llama_model.generate(**inputs, max_new_tokens=100)
    response_text = llama_tokenizer.decode(output[0], skip_special_tokens=True)

    # Update history
    with threading.Lock():
        sessions[session_id]["history"].extend([query, response_text])

    emit("query_response", {"response": response_text})
    print(f"Response sent: {response_text}")

# Start frame processing thread
def start_processing_threads():
    processing_threads = []
    for _ in range(MAX_WORKERS):
        thread = threading.Thread(target=frame_processing_worker)
        thread.daemon = True
        thread.start()
        processing_threads.append(thread)
    return processing_threads

if __name__ == "__main__":
    # Start processing threads
    threads = start_processing_threads()
    
    # Start server
    public_url = ngrok.connect(5000, "tcp")
    print("WebSocket Server URL:", public_url)
    
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)  # Set debug=False when using threads
    finally:
        # Shutdown gracefully
        processing_active = False
        
        # Add sentinel values to stop worker threads
        for _ in range(MAX_WORKERS):
            frame_queue.put(None)
            
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=5)