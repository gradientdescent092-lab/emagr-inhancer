import os
import uuid
import time
import cv2
import torch
from flask import Flask, render_template, request, send_from_directory, jsonify
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "RealESRGAN_x4plus.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_UPLOAD_MB = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# -----------------------------
# Load Real-ESRGAN once
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = device == "cuda"

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Put RealESRGAN_x4plus.pth in models/."
    )

upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=use_half,
    gpu_id=0 if use_half else None,
)

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route("/")
def index():
    return render_template(
        "index.html",
        device=device.upper(),
        scale="4×",
    )

@app.post("/enhance")
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Please choose an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Supported formats: PNG, JPG, JPEG, WEBP, BMP."
        }), 400

    job_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()

    input_name = f"{job_id}_input.{ext}"
    output_name = f"{job_id}_x4.png"

    input_path = os.path.join(UPLOAD_DIR, input_name)
    output_path = os.path.join(OUTPUT_DIR, output_name)

    file.save(input_path)

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "The uploaded image could not be decoded."}), 400

    input_width = img.shape[1]
    input_height = img.shape[0]

    start = time.time()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output, _ = upsampler.enhance(img, outscale=4)

        if output.dtype != "uint8":
            output = output.clip(0, 255).astype("uint8")

        if not cv2.imwrite(output_path, output):
            raise RuntimeError("Could not save the enhanced image.")

        elapsed = time.time() - start

    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return jsonify({
                "error": "GPU memory was insufficient. Try a smaller image."
            }), 500
        return jsonify({"error": str(exc)}), 500

    output_height, output_width = output.shape[:2]

    return jsonify({
        "success": True,
        "download_url": f"/download/{output_name}",
        "input_width": input_width,
        "input_height": input_height,
        "output_width": output_width,
        "output_height": output_height,
        "scale": "4×",
        "time_seconds": round(elapsed, 2),
    })

@app.get("/download/<filename>")
def download(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=True,
        download_name="enhanced_x4.png",
    )

@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "error": f"Image is too large. Maximum size is {MAX_UPLOAD_MB} MB."
    }), 413

if __name__ == "__main__":
    print("=" * 60)
    print("Real-ESRGAN Super Resolution Web App")
    print("=" * 60)
    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Scale: 4×")
    print("Tile: 256")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(host="127.0.0.1", port=5000, debug=False)
