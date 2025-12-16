#!/bin/bash
# Setup script for Reporter-AI environment
# This script sets up the conda environment with all dependencies using staged installation
# to avoid pip "resolution-too-deep" errors

set -e  # Exit on error

echo "=========================================="
echo "Reporter-AI Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "   Please install Miniconda or Anaconda first"
    exit 1
fi

# Check for FFmpeg (required for audio processing)
echo "🔍 Checking system dependencies..."
# Check in conda environment PATH
FFMPEG_FOUND=false
if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
    FFMPEG_FOUND=true
elif [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/bin/ffmpeg" ] && [ -f "${CONDA_PREFIX}/bin/ffprobe" ]; then
    FFMPEG_FOUND=true
fi

if [ "$FFMPEG_FOUND" = false ]; then
    echo "⚠️  Warning: FFmpeg not found. It's required for audio processing."
    echo "   Installing FFmpeg via conda..."
    
    # Install FFmpeg in the conda environment
    conda install -c conda-forge ffmpeg -y
    echo "✅ FFmpeg installed via conda"
    echo "   Note: The app will automatically configure pydub to use FFmpeg from the conda environment"
else
    echo "✅ FFmpeg found"
fi
echo ""

# Check if environment exists
ENV_NAME="reporter"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✅ Conda environment '${ENV_NAME}' already exists"
    echo "   Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
else
    echo "📦 Creating conda environment '${ENV_NAME}' with Python 3.10..."
    conda create -n ${ENV_NAME} python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

# Stage 1: Core numerical libraries (must be first)
echo ""
echo "📦 Stage 1: Core Numerical Libraries"
echo "----------------------------------------"
echo "   Upgrading NumPy (critical - must be done first)..."
pip install --upgrade --force-reinstall "numpy>=1.25.0,<2.0.0"
echo "   Installing NetworkX..."
pip install "networkx>=3.0"
echo "✅ Stage 1 complete"

# Stage 2: PyTorch ecosystem
echo ""
echo "📦 Stage 2: PyTorch Ecosystem"
echo "----------------------------------------"
echo "   (Using CPU version - modify script for CUDA if needed)"
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "torchdiffeq>=0.2.0" "torchsde>=0.2.0"
echo "✅ Stage 2 complete"

# Stage 3: TensorFlow (echomimic requirement)
# Must be installed AFTER numpy upgrade to ensure compatibility
echo ""
echo "📦 Stage 3: TensorFlow"
echo "----------------------------------------"
echo "   Reinstalling TensorFlow to ensure numpy compatibility..."
pip uninstall tensorflow tensorflow-estimator -y 2>/dev/null || true
pip install --no-cache-dir --force-reinstall "tensorflow==2.15.0"
echo "✅ Stage 3 complete"

# Stage 4: Transformers & HuggingFace
echo ""
echo "📦 Stage 4: Transformers & HuggingFace"
echo "----------------------------------------"
pip install "transformers>=4.46.2" "diffusers>=0.30.1" "accelerate>=0.25.0"
pip install "safetensors>=0.3.0" "huggingface-hub>=0.20.0,<1.0" "datasets>=2.16.1"
echo "✅ Stage 4 complete"

# Stage 5: Image & Video Processing
echo ""
echo "📦 Stage 5: Image & Video Processing"
echo "----------------------------------------"
pip install "Pillow>=9.0.0" "opencv-python>=4.5.0" "scikit-image>=0.20.0"
pip install "albumentations>=1.3.0" "imageio[ffmpeg]>=2.25.0" "imageio[pyav]>=2.25.0"
pip install "moviepy==2.2.1" "retina-face==0.0.17"
echo "✅ Stage 5 complete"

# Stage 6: Deep Learning Utilities
echo ""
echo "📦 Stage 6: Deep Learning Utilities"
echo "----------------------------------------"
pip install "einops>=0.6.0" "timm>=0.9.0" "tomesd>=0.1.0"
pip install "decord>=0.6.0" "tensorboard>=2.13.0" "onnxruntime>=1.15.0"
echo "✅ Stage 6 complete"

# Stage 7: Configuration & Utilities
echo ""
echo "📦 Stage 7: Configuration & Utilities"
echo "----------------------------------------"
pip install "omegaconf>=2.3.0" "SentencePiece>=0.1.99" "ftfy>=6.1.0"
pip install "func_timeout>=4.3.5" "beautifulsoup4>=4.12.0"
echo "✅ Stage 7 complete"

# Stage 8: TTS & Audio
echo ""
echo "📦 Stage 8: TTS & Audio Processing"
echo "----------------------------------------"
echo "   Note: TTS may warn about numpy version, but it works with numpy>=1.25.0"
pip install "TTS>=0.22.0" "pydub>=0.25.1" "soundfile>=0.12.0" "librosa>=0.10.0"
echo "✅ Stage 8 complete"

# Stage 9: Web Framework
echo ""
echo "📦 Stage 9: Web Framework"
echo "----------------------------------------"
pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0" "pydantic>=2.0.0"
echo "✅ Stage 9 complete"

# Stage 10: HTTP & Networking
echo ""
echo "📦 Stage 10: HTTP & Networking"
echo "----------------------------------------"
pip install "aiohttp>=3.9.0" "requests>=2.31.0"
echo "✅ Stage 10 complete"

# Stage 11: LLM API Clients
echo ""
echo "📦 Stage 11: LLM API Clients"
echo "----------------------------------------"
pip install "anthropic>=0.18.0" "openai>=1.0.0"
echo "✅ Stage 11 complete"

# Stage 12: Environment & Config
echo ""
echo "📦 Stage 12: Environment & Config"
echo "----------------------------------------"
pip install "python-dotenv>=1.0.0" "pyyaml>=6.0" "packaging>=21.0"
echo "✅ Stage 12 complete"

# Stage 13: Gradio & Memory Management
echo ""
echo "📦 Stage 13: Gradio & Memory Management"
echo "----------------------------------------"
pip install "gradio>=3.41.2" "mmgp>=0.1.0"
echo "✅ Stage 13 complete"

echo ""
echo "🧪 Running dependency tests..."
python test_dependencies.py

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run the application:"
echo "  conda activate ${ENV_NAME}"
echo "  python src/app/main.py"
echo ""

