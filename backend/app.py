import os
import sys
import json
import base64
import time
import asyncio
import threading
import subprocess
import glob
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif"}

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='./client/build')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
loaded_model = None
loaded_processor = None
current_model_name = None
model_loading = False

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
vram_limit = 8 * 1024 * 1024 * 1024  # 8GB in bytes

def get_available_models() -> List[str]:
    """Get list of available models in the models directory."""
    model_dirs = [d for d in os.listdir(MODEL_DIR) 
                 if os.path.isdir(os.path.join(MODEL_DIR, d))]
    
    # If no models are found, add a default one for demo purposes
    if not model_dirs:
        model_dirs = ["llava-v1.5-7b"]
        
    return model_dirs

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

def load_model(model_name: str) -> Dict[str, Any]:
    """Load a model from the models directory."""
    global loaded_model, loaded_processor, current_model_name, model_loading
    
    if model_loading:
        return {"success": False, "message": "Another model is currently loading"}
    
    model_loading = True
    current_model_name = model_name
    
    try:
        # Send initial progress update
        socketio.emit('model_status', {
            'type': 'model_status',
            'name': model_name,
            'loaded': False,
            'progress': 0
        })
        
        model_path = os.path.join(MODEL_DIR, model_name)
        
        # For demonstration, if the model doesn't exist, we'll simulate it
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            
            # Simulate model loading with progress updates
            for progress in range(0, 101, 10):
                socketio.emit('model_status', {
                    'type': 'model_status',
                    'name': model_name,
                    'loaded': False,
                    'progress': progress
                })
                time.sleep(0.5)  # Simulate loading time
                
            # For demonstration purposes, we're simulating a model
            # In a real app, you would use:
            # loaded_processor = AutoProcessor.from_pretrained(model_path)
            # loaded_model = AutoModelForVision2Seq.from_pretrained(
            #     model_path,
            #     torch_dtype=torch.float16,
            #     low_cpu_mem_usage=True,
            #     device_map="auto" if cuda_available else None
            # )
            
            # Mock the model and processor for demonstration
            class MockProcessor:
                def __call__(self, images, text=None, return_tensors="pt"):
                    return {"pixel_values": torch.rand(1, 3, 224, 224), "input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
            
            class MockModel:
                def __init__(self):
                    self.device = device
                    
                def generate(self, **kwargs):
                    return torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
                
                def to(self, device):
                    return self
            
            loaded_processor = MockProcessor()
            loaded_model = MockModel()
            
            # Final status update - model loaded
            socketio.emit('model_status', {
                'type': 'model_status',
                'name': model_name,
                'loaded': True,
                'progress': 100
            })
            
            return {"success": True, "message": f"Model {model_name} loaded successfully"}
        
        # Actual model loading logic
        try:
            # Progress update - 10%
            socketio.emit('model_status', {
                'type': 'model_status',
                'name': model_name, 
                'loaded': False,
                'progress': 10
            })
            
            # Load processor
            loaded_processor = AutoProcessor.from_pretrained(model_path)
            
            # Progress update - 30%
            socketio.emit('model_status', {
                'type': 'model_status',
                'name': model_name,
                'loaded': False, 
                'progress': 30
            })
            
            # Determine if we can fit in VRAM
            use_device_map = "auto"
            
            if cuda_available:
                # Check available VRAM
                total_vram = torch.cuda.get_device_properties(0).total_memory
                if total_vram <= vram_limit:
                    # Use CPU offloading if VRAM is limited
                    use_device_map = {"": "cpu"}
            
            # Progress update - 50%
            socketio.emit('model_status', {
                'type': 'model_status', 
                'name': model_name,
                'loaded': False,
                'progress': 50
            })
            
            # Load model with appropriate settings
            loaded_model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if cuda_available else torch.float32,
                low_cpu_mem_usage=True,
                device_map=use_device_map
            )
            
            # Progress updates 70% -> 100%
            for progress in range(70, 101, 10):
                socketio.emit('model_status', {
                    'type': 'model_status',
                    'name': model_name,
                    'loaded': progress == 100,
                    'progress': progress
                })
                time.sleep(0.2)  # Small delay for visual feedback
            
            return {"success": True, "message": f"Model {model_name} loaded successfully"}
        
        except Exception as e:
            return {"success": False, "message": f"Error loading model: {str(e)}"}
    
    finally:
        model_loading = False

def unload_model() -> Dict[str, Any]:
    """Unload the currently loaded model."""
    global loaded_model, loaded_processor, current_model_name
    
    if loaded_model is None:
        return {"success": False, "message": "No model is currently loaded"}
    
    try:
        # Release CUDA memory if using GPU
        if cuda_available and hasattr(loaded_model, 'to'):
            loaded_model.to('cpu')
            torch.cuda.empty_cache()
        
        loaded_model = None
        loaded_processor = None
        model_name = current_model_name
        current_model_name = None
        
        return {"success": True, "message": f"Model {model_name} unloaded successfully"}
    
    except Exception as e:
        return {"success": False, "message": f"Error unloading model: {str(e)}"}

def process_image(image: Image.Image) -> str:
    """Process an image with the loaded model."""
    global loaded_model, loaded_processor
    
    if loaded_model is None or loaded_processor is None:
        return "Error: No model is loaded"
    
    try:
        # Generate a text prompt
        prompt = "Describe what you see in this image in detail."
        
        # Prepare the image and prompt for the model
        inputs = loaded_processor(images=image, text=prompt, return_tensors="pt")
        
        # Move input tensors to the correct device
        for k, v in inputs.items():
            if hasattr(v, 'to') and cuda_available:
                inputs[k] = v.to(device)
        
        # Send initial progress update for processing
        socketio.emit('processing', {
            'type': 'processing',
            'progress': 0
        })
        
        # Generate prediction with progress updates
        # In a real implementation, you'd use the model's generate method
        # Here we'll simulate the progression
        
        with torch.no_grad():
            # Model inference
            output_ids = loaded_model.generate(
                **inputs,
                max_length=512,
                do_sample=False
            )
        
        # Send final progress update
        socketio.emit('processing', {
            'type': 'processing',
            'progress': 100
        })
        
        # Decode the output ids to text
        if hasattr(loaded_processor, 'batch_decode'):
            result = loaded_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            # Fallback for models without batch_decode
            result = "This is a detailed description of the image. [Sample output for demonstration]"
        
        return result
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Flask routes
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
    
    result = load_model(model_name)
    return jsonify(result)

@app.route("/api/model/unload", methods=["POST"])
def api_unload_model():
    """API endpoint to unload the current model."""
    result = unload_model()
    return jsonify(result)

@app.route("/api/process", methods=["POST"])
def api_process_image():
    """API endpoint to process an image."""
    if "image" not in request.json:
        return jsonify({"success": False, "message": "No image provided"})
    
    try:
        image_data = request.json["image"]
        image = preprocess_image(image_data)
        
        result = process_image(image)
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        })

# Socket.IO event handlers
@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    models = get_available_models()
    emit("models", {"type": "models", "models": models})

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")

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
        load_model(model_name)
    
    threading.Thread(target=load_model_thread).start()
    
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
    
    if loaded_model is None or loaded_processor is None:
        emit("error", {
            "type": "error",
            "message": "No model is loaded"
        })
        return
    
    try:
        image_data = data["image"]
        image = preprocess_image(image_data)
        
        # Start processing in a separate thread
        def process_thread():
            try:
                # Send processing status
                socketio.emit("processing", {
                    "type": "processing",
                    "progress": 10
                })
                
                # Process the image
                result = process_image(image)
                
                # Send result
                socketio.emit("result", {
                    "type": "result",
                    "result": result
                })
            
            except Exception as e:
                socketio.emit("error", {
                    "type": "error",
                    "message": f"Error processing image: {str(e)}"
                })
        
        threading.Thread(target=process_thread).start()
        
    except Exception as e:
        emit("error", {
            "type": "error",
            "message": f"Error processing image: {str(e)}"
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
        print(f"Starting localhost.run tunnel with command: {cmd}")
        print("This will create a HTTPS tunnel to your local app.")
        print("Look for a URL in the format https://[random]-localhost.run")
        
        # Read and print the output to find the URL
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.strip())
            
            # Look for the URL in the output
            if "tunneled with tls termination" in line:
                print("\n" + "="*50)
                print("HTTPS TUNNEL ACTIVE")
                print("Copy the URL above and open it in your mobile browser")
                print("="*50 + "\n")
        
        return process
    
    except Exception as e:
        print(f"Error setting up localhost.run: {e}")
        return None

if __name__ == "__main__":
    # Check for command line arguments
    port = 5000
    use_tunnel = "--tunnel" in sys.argv
    
    print(f"Starting Flask server on port {port}")
    
    # Start localhost.run tunnel if requested
    tunnel_process = None
    if use_tunnel:
        print("Setting up localhost.run tunnel for HTTPS access...")
        tunnel_process = setup_localhost_run(port)
    
    try:
        # Start the Flask app
        socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
    
    finally:
        # Clean up
        if tunnel_process:
            tunnel_process.terminate()
            print("Localhost.run tunnel terminated")