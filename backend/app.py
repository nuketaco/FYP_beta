#!/usr/bin/env python
# --- IMPORTANT: Monkey patch MUST be done first, before any other imports ---
import eventlet
eventlet.monkey_patch()

# --- Standard library imports ---
import os
import sys
import json
import base64
import time
import threading
import subprocess
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Flask and related imports ---
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# --- Create app and configure Socket.IO before other imports ---
app = Flask(__name__, static_folder='./client/build')
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25
)

# --- ML-related imports ---
# These come after Flask and Socket.IO setup to avoid context issues
from PIL import Image
import torch

# --- Configuration constants ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif"}
VRAM_LIMIT = 8.5 * 1024 * 1024 * 1024  # 8.5GB in bytes

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global state management ---
class AppState:
    def __init__(self):
        self.loaded_model = None
        self.loaded_processor = None
        self.current_model_name = None
        self.model_loading = False
        self.active_processing_tasks = {}  # Map of processing_id -> cancel_flag
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        
        if self.cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU: {gpu_name} with {vram:.2f} GB VRAM")
            except Exception as e:
                logger.warning(f"Error getting GPU info: {e}")

# Initialize state
state = AppState()

# --- Helper functions ---
def get_available_models() -> List[str]:
    """Get list of available models in the models directory."""
    try:
        model_dirs = [d for d in os.listdir(MODEL_DIR) 
                    if os.path.isdir(os.path.join(MODEL_DIR, d))]
        logger.info(f"Looking for models in: {os.path.abspath(MODEL_DIR)}")
        logger.info(f"Found models: {model_dirs}")
        return model_dirs
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data: str) -> Image.Image:
    """Process base64 image data into PIL Image."""
    # Remove data URL prefix if present
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    return image

def resize_image_for_processing(image, max_size=512):
    """Resize an image to limit maximum dimension while preserving aspect ratio."""
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image  # No need to resize
    
    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize and return
    try:
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    except AttributeError:
        # Fallback for older Pillow versions that might not have LANCZOS
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    return resized_image

def cancel_processing(processing_id: str) -> bool:
    """Cancel an in-progress image processing task."""
    if processing_id not in state.active_processing_tasks:
        return False
    
    # Set the cancel flag
    state.active_processing_tasks[processing_id]['cancelled'] = True
    return True

# --- Model management functions ---
def load_model(model_name: str) -> Dict[str, Any]:
    """Load a ResNet model for image recognition."""
    if state.model_loading:
        return {"success": False, "message": "Another model is currently loading"}
    
    state.model_loading = True
    state.current_model_name = model_name
    
    try:
        # Import the necessary libraries
        from transformers import AutoFeatureExtractor, ResNetForImageClassification
        
        # Send initial progress update
        socketio.emit('model_status', {
            'type': 'model_status',
            'name': model_name,
            'loaded': False,
            'progress': 0
        })
        
        model_path = os.path.join(MODEL_DIR, model_name)
        
        if not os.path.exists(model_path):
            return {"success": False, "message": f"Model path {model_path} does not exist"}
            
        # Progress update - 10%
        socketio.emit('model_status', {
            'type': 'model_status',
            'name': model_name, 
            'loaded': False,
            'progress': 10
        })
        
        # Load feature extractor (processor)
        try:
            state.loaded_processor = AutoFeatureExtractor.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            return {"success": False, "message": f"Error loading feature extractor: {str(e)}"}
        
        # Progress update - 30%
        socketio.emit('model_status', {
            'type': 'model_status',
            'name': model_name,
            'loaded': False, 
            'progress': 30
        })
        
        # Progress update - 50%
        socketio.emit('model_status', {
            'type': 'model_status', 
            'name': model_name,
            'loaded': False,
            'progress': 50
        })
        
        # Load model with memory-efficient settings
        try:
            state.loaded_model = ResNetForImageClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use float32 for better compatibility
                low_cpu_mem_usage=True
            )
            
            # Move to CPU to start - we'll move specific operations to GPU as needed
            state.loaded_model.to("cpu")
            
            # Clear GPU memory
            if state.cuda_available:
                torch.cuda.empty_cache()
            
        except Exception as e:
            state.loaded_processor = None
            logger.error(f"Error loading model: {e}")
            return {"success": False, "message": f"Error loading model: {str(e)}"}
        
        # Progress updates 70% -> 100%
        for progress in range(70, 101, 10):
            socketio.emit('model_status', {
                'type': 'model_status',
                'name': model_name,
                'loaded': progress == 100,
                'progress': progress,
                'cuda_available': state.cuda_available
            })
            eventlet.sleep(0.2)
        
        return {"success": True, "message": f"ResNet model {model_name} loaded successfully"}
    
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        return {"success": False, "message": f"Error loading model: {str(e)}"}
    
    finally:
        state.model_loading = False

def unload_model() -> Dict[str, Any]:
    """Unload the currently loaded model."""
    if state.loaded_model is None:
        return {"success": False, "message": "No model is currently loaded"}
    
    try:
        # Release CUDA memory if using GPU
        if state.cuda_available and hasattr(state.loaded_model, 'to'):
            state.loaded_model.to('cpu')
            torch.cuda.empty_cache()
        
        model_name = state.current_model_name
        state.loaded_model = None
        state.loaded_processor = None
        state.current_model_name = None
        
        return {"success": True, "message": f"Model {model_name} unloaded successfully"}
    
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        return {"success": False, "message": f"Error unloading model: {str(e)}"}

def update_progress(step, output, processing_id, cancel_flag):
    """Update progress during model generation and check for cancellation."""
    max_steps = 100  # Typical model generation steps
    progress = min(int((step / max_steps) * 100), 75)  # Cap at 75% for generation phase
    
    # Emit progress update every few steps to avoid flooding
    if step % 5 == 0 or step == 1:
        socketio.emit('processing', {
            'type': 'processing',
            'progress': progress,
            'processing_id': processing_id
        })
    
    # Check if this task has been cancelled
    return cancel_flag['cancelled']

def process_image(image: Image.Image, processing_id: str) -> str:
    """Process an image with the loaded model."""
    if state.loaded_model is None or state.loaded_processor is None:
        return "Error: No model is loaded"
    
    # Initialize a flag for this processing task
    cancel_flag = {'cancelled': False}
    state.active_processing_tasks[processing_id] = cancel_flag
    
    try:
        # Clear CUDA cache before processing
        if state.cuda_available:
            torch.cuda.empty_cache()
            
        # Resize image to reduce memory usage
        image = resize_image_for_processing(image, max_size=512)
        
        # Generate a text prompt
        prompt = "Describe what you see in this image in detail."
        
        # Prepare the image and prompt for the model
        inputs = state.loaded_processor(images=image, text=prompt, return_tensors="pt")
        
        # Move input tensors to the correct device
        for k, v in inputs.items():
            if hasattr(v, 'to') and state.cuda_available:
                inputs[k] = v.to(state.device)
        
        # Send initial progress update for processing
        socketio.emit('processing', {
            'type': 'processing',
            'progress': 10,
            'processing_id': processing_id
        })
        
        # Check if cancelled before generation
        if cancel_flag['cancelled']:
            return "Processing cancelled by user"
        
        # Generate prediction
        with torch.no_grad():
            # Model inference with progress updates
            # Note: For some models, the callback may not be supported
            try:
                output_ids = state.loaded_model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced token count
                    do_sample=False,
                    callback=lambda step, output, params: update_progress(step, output, processing_id, cancel_flag)
                )
            except TypeError:
                # Fallback if callback is not supported
                logger.info("Model doesn't support callback, using simple generation")
                # Emit progress updates manually
                socketio.emit('processing', {
                    'type': 'processing',
                    'progress': 25,
                    'processing_id': processing_id
                })
                
                output_ids = state.loaded_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
                
                socketio.emit('processing', {
                    'type': 'processing',
                    'progress': 50,
                    'processing_id': processing_id
                })
            
            # Update progress during generation
            socketio.emit('processing', {
                'type': 'processing',
                'progress': 75,
                'processing_id': processing_id
            })
            
            # Check if cancelled after generation
            if cancel_flag['cancelled']:
                return "Processing cancelled by user"
        
        # Send final progress update
        socketio.emit('processing', {
            'type': 'processing',
            'progress': 100,
            'processing_id': processing_id
        })
        
        # Decode the output ids to text
        if hasattr(state.loaded_processor, 'batch_decode'):
            result = state.loaded_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            # Fallback for models without batch_decode
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, state.current_model_name))
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Clear CUDA cache after processing
        if state.cuda_available:
            torch.cuda.empty_cache()
            
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"Error processing image: {str(e)}"
    
    finally:
        # Remove from active tasks
        if processing_id in state.active_processing_tasks:
            del state.active_processing_tasks[processing_id]

# --- Flask routes ---
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """Serve the React frontend."""
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route("/api/models", methods=["GET"])
def api_get_models():
    """API endpoint to get available models."""
    models = get_available_models()
    return jsonify({"models": models})

@app.route("/api/model/load", methods=["POST"])
def api_load_model():
    """API endpoint to load a model."""
    data = request.json
    model_name = data.get("model")
    
    if not model_name:
        return jsonify({"success": False, "message": "Model name is required"})
    
    # Start model loading in a separate thread to not block the server
    def load_model_thread():
        result = load_model(model_name)
        if not result["success"]:
            logger.error(f"Error loading model: {result['message']}")
    
    thread = threading.Thread(target=load_model_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "message": f"Loading model {model_name}..."})

@app.route("/api/model/unload", methods=["POST"])
def api_unload_model():
    """API endpoint to unload the current model."""
    result = unload_model()
    return jsonify(result)

@app.route("/api/process-image-v2", methods=["POST"])
def api_process_image_resnet():
    """REST API endpoint for image recognition with ResNet."""
    if "image" not in request.json:
        return jsonify({"success": False, "message": "No image provided"})
    
    try:
        # Clear CUDA cache before processing
        if state.cuda_available:
            torch.cuda.empty_cache()
            
        # Get image data
        image_data = request.json["image"]
        
        # Process image
        image = preprocess_image(image_data)
        image = resize_image_for_processing(image, max_size=384)  # Lower resolution for memory efficiency
        
        if state.loaded_model is None or state.loaded_processor is None:
            return jsonify({"success": False, "message": "No model is loaded"})
        
        # ResNet-specific processing
        try:
            # Prepare the image
            inputs = state.loaded_processor(images=image, return_tensors="pt")
            
            # Process on CPU first
            with torch.no_grad():
                # Optional: Move to GPU just for inference if there's enough memory
                if state.cuda_available:
                    # Get a chunk of GPU that we can fit in memory
                    try:
                        # Move model temporarily to GPU for faster inference
                        state.loaded_model.to("cuda")
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        use_cuda = True
                    except Exception as e:
                        logger.warning(f"Could not use CUDA for inference: {e}")
                        state.loaded_model.to("cpu")
                        inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        use_cuda = False
                else:
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    use_cuda = False
                
                # Run inference
                logger.info(f"Running inference on: {'CUDA' if use_cuda else 'CPU'}")
                outputs = state.loaded_model(**inputs)
                
                # Move model back to CPU to save GPU memory
                if use_cuda:
                    state.loaded_model.to("cpu")
                    torch.cuda.empty_cache()
                
            # Process the logits
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
            # Get the label for the predicted class
            if hasattr(state.loaded_model.config, "id2label"):
                result = state.loaded_model.config.id2label[predicted_class_idx]
            else:
                result = f"Class {predicted_class_idx}"
                
            # Get top 5 predictions for more detailed results
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            top5_results = []
            for i, (prob, idx) in enumerate(zip(top5_probs.tolist(), top5_indices.tolist())):
                if hasattr(state.loaded_model.config, "id2label"):
                    label = state.loaded_model.config.id2label[idx]
                else:
                    label = f"Class {idx}"
                
                top5_results.append(f"{label}: {prob*100:.2f}%")
            
            # Combine into a more detailed result
            result = f"Top prediction: {result}\n\nTop 5 predictions:\n" + "\n".join(top5_results)
                
        except Exception as e:
            logger.error(f"ResNet processing failed: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "message": f"Error processing with ResNet model: {str(e)}"
            })
        
        # Clear CUDA cache after processing
        if state.cuda_available:
            torch.cuda.empty_cache()
            
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"REST API process error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        })

@app.route("/api/process", methods=["POST"])
def api_process_image():
    """API endpoint to process an image."""
    if "image" not in request.json:
        return jsonify({"success": False, "message": "No image provided"})
    
    try:
        image_data = request.json["image"]
        processing_id = str(uuid.uuid4())
        image = preprocess_image(image_data)
        
        # Start processing in a separate thread
        def process_thread():
            try:
                result = process_image(image, processing_id)
                
                # Check if processing was cancelled
                if result == "Processing cancelled by user":
                    return
                
                # Send result
                socketio.emit("result", {
                    "type": "result",
                    "result": result,
                    "processing_id": processing_id
                })
            except Exception as e:
                logger.error(f" critical Error in process thread: {e}", exc_info=True)
                
                try:
                    socketio.emit("error", {
                        "type": "error",
                        "message": f"Error processing image: {str(e)}",
                        "processing_id": processing_id
                })
                except:
                    logger.error("could not notify client of error")
        
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Processing started",
            "processing_id": processing_id
        })
    
    except Exception as e:
        logger.error(f"API process error: {e}")
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        })

@app.route("/api/cancel", methods=["POST"])
def api_cancel_processing():
    """API endpoint to cancel image processing."""
    data = request.json
    processing_id = data.get("processing_id")
    
    if not processing_id:
        return jsonify({"success": False, "message": "Processing ID is required"})
    
    result = cancel_processing(processing_id)
    
    return jsonify({
        "success": result,
        "message": "Processing cancelled" if result else "No active processing found with this ID"
    })

# --- Socket.IO event handlers ---
@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")
    models = get_available_models()
    emit("models", {"type": "models", "models": models})

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")

@socketio.on("get_models")
def handle_get_models():
    """Send available models to the client."""
    models = get_available_models()
    emit("models", {"type": "models", "models": models})

@socketio.on("load_model")
def handle_load_model(data):
    """Handle model loading request."""
    model_name = data.get("model")
    
    if not model_name:
        emit("error", {
            "type": "error",
            "message": "Model name is required"
        })
        return
    
    # Start model loading in a separate thread to not block the server
    def load_model_thread():
        result = load_model(model_name)
        if not result["success"]:
            socketio.emit("error", {
                "type": "error",
                "message": result["message"]
            })
    
    thread = threading.Thread(target=load_model_thread)
    thread.daemon = True
    thread.start()
    
    emit("model_status", {
        "type": "model_status",
        "name": model_name,
        "loaded": False,
        "progress": 0,
        "message": f"Loading model {model_name}..."
    })

@socketio.on("unload_model")
def handle_unload_model():
    """Handle model unloading request."""
    result = unload_model()
    
    if result["success"]:
        emit("model_status", {
            "type": "model_status",
            "loaded": False,
            "progress": 0
        })
    else:
        emit("error", {
            "type": "error",
            "message": result["message"]
        })

@socketio.on("process_image")
def handle_process_image(data):
    """Handle image processing request."""
    if "image" not in data:
        emit("error", {
            "type": "error",
            "message": "No image provided"
        })
        return
    
    if state.loaded_model is None or state.loaded_processor is None:
        emit("error", {
            "type": "error",
            "message": "No model is loaded"
        })
        return
    
    try:
        image_data = data["image"]
        image = preprocess_image(image_data)
        
        # Generate a unique ID for this processing task
        processing_id = str(uuid.uuid4())
        
        # Start processing in a separate thread
        def process_thread():
            try:
                # Send processing status with the processing ID
                socketio.emit("processing", {
                    "type": "processing",
                    "progress": 5,
                    "processing_id": processing_id
                })
                
                # Process the image
                result = process_image(image, processing_id)
                
                # Check if processing was cancelled
                if result == "Processing cancelled by user":
                    socketio.emit("cancelled", {
                        "type": "cancelled",
                        "processing_id": processing_id
                    })
                    return
                
                # Send result
                socketio.emit("result", {
                    "type": "result",
                    "result": result,
                    "processing_id": processing_id
                })
            
            except Exception as e:
                logger.error(f"Socket process error: {e}")
                socketio.emit("error", {
                    "type": "error",
                    "message": f"Error processing image: {str(e)}",
                    "processing_id": processing_id
                })
        
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
        
        # Send the processing ID to the client immediately
        emit("processing", {
            "type": "processing",
            "progress": 0,
            "processing_id": processing_id,
            "message": "Starting image processing..."
        })
        
    except Exception as e:
        logger.error(f"Socket image processing error: {e}")
        emit("error", {
            "type": "error",
            "message": f"Error processing image: {str(e)}"
        })

@socketio.on("cancel_processing")
def handle_cancel_processing(data):
    """Handle request to cancel image processing."""
    processing_id = data.get("processing_id")
    
    if not processing_id:
        emit("error", {
            "type": "error",
            "message": "Processing ID is required"
        })
        return
    
    result = cancel_processing(processing_id)
    
    if result:
        emit("processing", {
            "type": "processing",
            "progress": 0,
            "processing_id": processing_id,
            "message": "Cancelling processing..."
        })
    else:
        emit("error", {
            "type": "error",
            "message": "No active processing found with this ID"
        })

def setup_localhost_run(port=5000):
    """Setup localhost.run tunnel for HTTPS access."""
    try:
        # Use subprocess to start localhost.run
        cmd = f"ssh -R 80:localhost:{port} localhost.run"
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print command to start the tunnel
        logger.info(f"Starting localhost.run tunnel with command: {cmd}")
        logger.info("This will create a HTTPS tunnel to your local app.")
        logger.info("Look for a URL in the format https://[random]-localhost.run")
        
        # Read and print the output to find the URL
        while True:
            line = process.stdout.readline()
            if not line:
                break
            logger.info(line.strip())
            
            # Look for the URL in the output
            if "tunneled with tls termination" in line:
                logger.info("\n" + "="*50)
                logger.info("HTTPS TUNNEL ACTIVE")
                logger.info("Copy the URL above and open it in your mobile browser")
                logger.info("="*50 + "\n")
        
        return process
    
    except Exception as e:
        logger.error(f"Error setting up localhost.run: {e}")
        return None

# --- Main entry point ---
if __name__ == "__main__":
    # Check for command line arguments
    port = 5000
    use_tunnel = "--tunnel" in sys.argv
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"CUDA available: {state.cuda_available}")
    
    # Display available models
    models = get_available_models()
    if models:
        logger.info(f"Available models: {', '.join(models)}")
    else:
        logger.info(f"No models found in the models directory. Please add models to: {MODEL_DIR}")
    
    # Start localhost.run tunnel if requested
    tunnel_process = None
    if use_tunnel:
        logger.info("Setting up localhost.run tunnel for HTTPS access...")
        tunnel_process = setup_localhost_run(port)
    
    try:
        # Start the Flask app
        socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
    
    finally:
        # Clean up
        if tunnel_process:
            tunnel_process.terminate()
            logger.info("Localhost.run tunnel terminated")