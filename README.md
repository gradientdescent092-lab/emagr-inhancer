# Real-ESRGAN Super Resolution Web Application

A portfolio-ready web application that uses **Real-ESRGAN x4** to enhance uploaded images.

## What it does

1. User uploads an image.
2. Flask receives the image.
3. OpenCV decodes it.
4. Real-ESRGAN RRDB model performs 4× super-resolution.
5. Tile-based inference reduces GPU memory usage.
6. The enhanced PNG is returned to the browser.
7. The UI shows original vs enhanced resolution and inference time.
8. The user can download the enhanced image.

## Project structure

```text
RealESRGAN_WebApp/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── RealESRGAN_x4plus.pth
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── uploads/
└── outputs/
```

## 1. Create environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install PyTorch

Install the PyTorch build appropriate for your NVIDIA CUDA setup from the official PyTorch instructions.

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

If the Real-ESRGAN package is unavailable through your package index, install the source package:

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -e . --no-deps
cd ..
```

## 3. Add the model

Put:

```text
RealESRGAN_x4plus.pth
```

inside:

```text
models/
```

The model file is intentionally not committed to GitHub.

## 4. Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## GPU

If CUDA is available, the app uses GPU 0 and FP16 inference.

The app uses:

```text
tile = 256
tile_pad = 10
```

to reduce GPU memory usage.

If CUDA is unavailable, it falls back to CPU inference.

## Notes

This application performs inference only; it does not train Real-ESRGAN.

For production deployment, add authentication/rate limiting, persistent job storage, cleanup of old files, and a suitable GPU server.
