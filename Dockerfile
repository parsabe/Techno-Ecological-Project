# 🌊 AquaPulse Enterprise AI Vision & EnKF Telemetry Engine Docker Container
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install system runtime dependencies (OpenCV, FFmpeg, MiKTeX/TeXLive LaTeX engine)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement manifests and install Python dependencies
COPY requirements.txt req.txt /app/
RUN pip install --no-cache-dir -r requirements.txt -r req.txt

# Copy application source code, models, and assets
COPY . /app

# Default environment configurations for headless containerized execution
ENV HEADLESS=1
ENV PYTHONUNBUFFERED=1

# Entrypoint for running AquaPulse Neural Telemetry Engine in headless mode
ENTRYPOINT ["python", "3 - AI process/main.py", "--headless"]
CMD ["--help"]
