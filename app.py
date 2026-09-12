import os
import uuid
import time
import gc

import cv2
import torch

from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "RealESRGAN_x4plus.pth"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp"
}

MAX_UPLOAD_MB = 20

# ------------------------------------------------------------
# CPU DEPLOYMENT SETTINGS
# ------------------------------------------------------------

# Render Free has very limited CPU and RAM.
# Keep the inference image small.
MAX_INPUT_DIMENSION = 256

# Smaller tile reduces peak memory usage.
TILE_SIZE = 64

TILE_PAD = 10

# Limit PyTorch CPU thread usage.
# This reduces memory pressure on small servers.
CPU_THREADS = 1


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)

# Allow requests from Vercel frontend
CORS(app)


# ============================================================
# PYTORCH CPU CONFIGURATION
# ============================================================

if not torch.cuda.is_available():

    torch.set_num_threads(
        CPU_THREADS
    )

    try:

        torch.set_num_interop_threads(
            CPU_THREADS
        )

    except RuntimeError:
        pass


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "RealESRGAN_x4plus.pth not found. "
        "Make sure the model is inside models/."
    )


# ============================================================
# LOAD REAL-ESRGAN
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_half = (
    device == "cuda"
)


print("=" * 60)
print("Real-ESRGAN Backend")
print("=" * 60)

print(
    "Device:",
    device
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Scale: 4x"
)

print(
    "Tile:",
    TILE_SIZE
)

print(
    "Tile Padding:",
    TILE_PAD
)

print(
    "Maximum Input Dimension:",
    MAX_INPUT_DIMENSION
)

print(
    "CPU Threads:",
    CPU_THREADS
)

print("=" * 60)


# ============================================================
# CREATE RRDB NETWORK
# ============================================================

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# CREATE REAL-ESRGAN UPSAMPLER
# ============================================================

upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=TILE_SIZE,
    tile_pad=TILE_PAD,
    pre_pad=0,
    half=use_half,
    gpu_id=0 if use_half else None
)


print(
    "Real-ESRGAN model loaded successfully."
)

print("=" * 60)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():

    return jsonify({

        "status": "online",

        "service":
            "Real-ESRGAN Image Enhancement API",

        "device":
            device,

        "scale":
            "4x",

        "tile":
            TILE_SIZE,

        "max_input_dimension":
            MAX_INPUT_DIMENSION

    })


# ============================================================
# IMAGE ENHANCEMENT API
# ============================================================

@app.post("/enhance")
def enhance():

    # ----------------------------------------
    # Check uploaded image
    # ----------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image was uploaded."

        }), 400


    file = request.files["image"]


    if not file.filename:

        return jsonify({

            "success": False,

            "error":
                "Please choose an image."

        }), 400


    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "success": False,

            "error":
                (
                    "Supported formats: "
                    "PNG, JPG, JPEG, WEBP, BMP."
                )

        }), 400


    # ----------------------------------------
    # Generate unique filenames
    # ----------------------------------------

    job_id = uuid.uuid4().hex


    ext = file.filename.rsplit(
        ".",
        1
    )[1].lower()


    input_name = (
        f"{job_id}_input.{ext}"
    )

    output_name = (
        f"{job_id}_x4.png"
    )


    input_path = os.path.join(
        UPLOAD_DIR,
        input_name
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    # ----------------------------------------
    # Save uploaded image
    # ----------------------------------------

    file.save(
        input_path
    )


    # ----------------------------------------
    # Read image
    # ----------------------------------------

    img = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )


    if img is None:

        return jsonify({

            "success": False,

            "error":
                (
                    "The uploaded image "
                    "could not be decoded."
                )

        }), 400


    # ----------------------------------------
    # Original dimensions
    # ----------------------------------------

    original_width = (
        img.shape[1]
    )

    original_height = (
        img.shape[0]
    )


    # ----------------------------------------
    # Resize image for CPU deployment
    # ----------------------------------------

    max_dimension = max(
        original_width,
        original_height
    )


    resized = False


    if (
        max_dimension
        > MAX_INPUT_DIMENSION
    ):

        resize_scale = (
            MAX_INPUT_DIMENSION
            / max_dimension
        )


        new_width = max(
            1,
            int(
                original_width
                * resize_scale
            )
        )


        new_height = max(
            1,
            int(
                original_height
                * resize_scale
            )
        )


        img = cv2.resize(
            img,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )


        resized = True


    # ----------------------------------------
    # Inference dimensions
    # ----------------------------------------

    input_width = (
        img.shape[1]
    )

    input_height = (
        img.shape[0]
    )


    print("=" * 60)

    print(
        "New enhancement request"
    )

    print(
        "Original:",
        original_width,
        "x",
        original_height
    )

    print(
        "Inference:",
        input_width,
        "x",
        input_height
    )

    print(
        "Resized:",
        resized
    )

    print("=" * 60)


    # ----------------------------------------
    # Run Real-ESRGAN
    # ----------------------------------------

    start = time.time()


    try:

        # ------------------------------------
        # Clear unused Python memory
        # ------------------------------------

        gc.collect()


        # ------------------------------------
        # Clear CUDA cache if available
        # ------------------------------------

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        # ------------------------------------
        # Inference
        # ------------------------------------

        with torch.inference_mode():

            output, _ = (
                upsampler.enhance(
                    img,
                    outscale=4
                )
            )


        # ------------------------------------
        # Make sure output is uint8
        # ------------------------------------

        if output.dtype != "uint8":

            output = output.clip(
                0,
                255
            ).astype(
                "uint8"
            )


        # ------------------------------------
        # Save enhanced image
        # ------------------------------------

        success = cv2.imwrite(
            output_path,
            output
        )


        if not success:

            raise RuntimeError(
                "Could not save the enhanced image."
            )


        elapsed = (
            time.time()
            - start
        )


    except RuntimeError as exc:

        error_message = str(exc)


        # ------------------------------------
        # CUDA memory error
        # ------------------------------------

        if (
            "CUDA out of memory"
            in error_message
        ):

            if torch.cuda.is_available():

                torch.cuda.empty_cache()


            gc.collect()


            return jsonify({

                "success": False,

                "error":
                    (
                        "GPU memory was insufficient. "
                        "Try a smaller image."
                    )

            }), 500


        # ------------------------------------
        # Other RuntimeError
        # ------------------------------------

        gc.collect()


        return jsonify({

            "success": False,

            "error":
                error_message

        }), 500


    except Exception as exc:

        gc.collect()


        return jsonify({

            "success": False,

            "error":
                str(exc)

        }), 500


    # ----------------------------------------
    # Output dimensions
    # ----------------------------------------

    output_height = (
        output.shape[0]
    )

    output_width = (
        output.shape[1]
    )


    # ----------------------------------------
    # Create public download URL
    # ----------------------------------------

    download_url = (

        request.host_url.rstrip("/")

        + "/download/"

        + output_name
    )


    # ----------------------------------------
    # Cleanup input image
    # ----------------------------------------

    try:

        if os.path.exists(
            input_path
        ):

            os.remove(
                input_path
            )

    except Exception:

        pass


    # ----------------------------------------
    # Cleanup memory
    # ----------------------------------------

    del img
    del output

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ----------------------------------------
    # Response
    # ----------------------------------------

    return jsonify({

        "success": True,

        "download_url":
            download_url,

        "original_width":
            original_width,

        "original_height":
            original_height,

        "input_width":
            input_width,

        "input_height":
            input_height,

        "output_width":
            output_width,

        "output_height":
            output_height,

        "scale":
            "4x",

        "time_seconds":
            round(
                elapsed,
                2
            )

    })


# ============================================================
# DOWNLOAD ENHANCED IMAGE
# ============================================================

@app.get(
    "/download/<filename>"
)
def download(filename):

    return send_from_directory(

        OUTPUT_DIR,

        filename,

        as_attachment=True,

        download_name=
            "enhanced_x4.png"
    )


# ============================================================
# FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def too_large(_):

    return jsonify({

        "success": False,

        "error":
            (
                f"Image is too large. "
                f"Maximum size is "
                f"{MAX_UPLOAD_MB} MB."
            )

    }), 413


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Real-ESRGAN Super Resolution API"
    )

    print("=" * 60)

    print(
        "Device:",
        device
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Scale: 4x"
    )

    print(
        "Tile:",
        TILE_SIZE
    )

    print(
        "Maximum Input Dimension:",
        MAX_INPUT_DIMENSION
    )

    print(
        "CPU Threads:",
        CPU_THREADS
    )

    print(
        "Local URL: "
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(
    host="0.0.0.0",
    port=5000,
    debug=False
)