import os
import sys
import shutil
import subprocess

def force_remove_dir(dir_path):
    if not os.path.exists(dir_path):
        return
    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, 0o777)
            func(path)
        except Exception:
            pass
    try:
        shutil.rmtree(dir_path, onerror=_onerror)
    except Exception:
        pass

def create_docker_bundle(docker_dir, script_dir):
    os.makedirs(docker_dir, exist_ok=True)
    
    # 1. Dockerfile
    dockerfile_content = """# AquaPulse Enterprise AI Vision & Stochastic Telemetry Container
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ffmpeg \\
    libsm6 \\
    libxext6 \\
    libgl1-mesa-glx \\
    texlive-latex-base \\
    texlive-latex-extra \\
    texlive-fonts-recommended \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
WORKDIR /app/src

EXPOSE 8080 7860

ENTRYPOINT ["python", "main.py"]
"""
    with open(os.path.join(docker_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    # 2. docker-compose.yml
    compose_content = """version: '3.8'

services:
  aquapulse:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aquapulse_ai_engine
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./video_analysis_sessions:/app/src/video_analysis_sessions
      - ./input_videos:/app/src/input_videos
    restart: unless-stopped
"""
    with open(os.path.join(docker_dir, "docker-compose.yml"), "w", encoding="utf-8") as f:
        f.write(compose_content)

    # 3. requirements.txt
    reqs_content = """ultralytics>=8.1.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
scikit-learn>=1.2.0
requests>=2.31.0
gdown>=4.7.0
"""
    with open(os.path.join(docker_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(reqs_content)

    # 4. run_docker.bat
    bat_content = """@echo off
echo ========================================================
echo  AquaPulse Production Docker Deployment Launcher
echo ========================================================
echo Building Docker image 'aquapulse:latest'...
docker build -t aquapulse:latest .
echo Starting AquaPulse AI Engine via Docker Compose...
docker-compose up -d
echo AquaPulse Docker container is running!
pause
"""
    with open(os.path.join(docker_dir, "run_docker.bat"), "w", encoding="utf-8") as f:
        f.write(bat_content)

    # 5. README_DOCKER_DEPLOYMENT.md
    readme_docker = """# AquaPulse Production Docker Deployment Guide

This directory contains the self-contained production Docker bundle for deploying the **AquaPulse AI Neural Vision & Stochastic Telemetry Framework**.

## Directory Contents
- `Dockerfile`: Production CUDA 12.1 + PyTorch 2.2.1 + TeX Live container setup.
- `docker-compose.yml`: Multi-container orchestration definition with NVIDIA GPU pass-through.
- `requirements.txt`: Python package dependencies.
- `run_docker.bat`: Quick-start batch file for building and running on Docker Desktop.
- `src/`: Complete source application tree (models, core modules, assets).

## Quick Start (Docker Desktop / Linux GPU Server)

### 1. Build and Launch via Docker Compose:
```bash
docker-compose up --build -d
```

### 2. Check Container Logs:
```bash
docker logs -f aquapulse_ai_engine
```
"""
    with open(os.path.join(docker_dir, "README_DOCKER_DEPLOYMENT.md"), "w", encoding="utf-8") as f:
        f.write(readme_docker)

    # 6. Copy isolated source code to src/ inside Docker bundle
    docker_src = os.path.join(docker_dir, "src")
    force_remove_dir(docker_src)
    os.makedirs(docker_src, exist_ok=True)

    files_to_copy = [
        "main.py", "mod_00_config_and_assets.py", "mod_01_eco_census.py",
        "mod_02_stochastic_enkf.py", "mod_03_chart_renderer.py", "mod_04_vision_engine.py",
        "mod_05_dialogue_and_ollama.py", "mod_06_ui_dashboard.py", "mod_07_pdf_exporter.py",
        "manual_botsort.py", "johnny.gif", "report_template.tex"
    ]
    for fn in files_to_copy:
        src_f = os.path.join(script_dir, fn)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(docker_src, fn))

    # Copy models/ folder into Docker src/
    models_src = os.path.join(script_dir, "models")
    if os.path.exists(models_src):
        shutil.copytree(models_src, os.path.join(docker_src, "models"), dirs_exist_ok=True)

def main():
    script_dir = os.path.abspath(r"C:\Users\parsa\Desktop\Code\3 - AI process")
    install_win_dir = r"C:\Users\parsa\Desktop\Install\windows"
    install_docker_dir = r"C:\Users\parsa\Desktop\Install\Docker"
    
    os.makedirs(install_win_dir, exist_ok=True)
    os.makedirs(install_docker_dir, exist_ok=True)
    
    pyinstaller_exe = r"C:\Users\parsa\Desktop\Code\venv\Scripts\pyinstaller.exe"
    if not os.path.exists(pyinstaller_exe):
        pyinstaller_exe = shutil.which("pyinstaller") or "pyinstaller"

    print("==========================================================")
    print(" [Build] AquaPulse Windows Setup & Docker Deployment Pipeline")
    print("==========================================================")

    dist_app = os.path.join(script_dir, "dist", "AquaPulse_App")
    force_remove_dir(dist_app)

    # 1. Build AquaPulse_App
    print("\n[Step 1/4] Building AquaPulse Application package (AquaPulse_App)...")
    spec_app = os.path.join(script_dir, "AquaPulse.spec")
    subprocess.run([pyinstaller_exe, spec_app, "--noconfirm"], cwd=script_dir, check=True)

    if not os.path.exists(dist_app):
        print(f"[ERROR] Compiled application bundle not found at {dist_app}")
        sys.exit(1)

    # 2. Build Uninstall.exe
    print("\n[Step 2/4] Building Control Panel Uninstaller (Uninstall.exe)...")
    spec_uninstall = os.path.join(script_dir, "AquaPulse_Uninstaller.spec")
    subprocess.run([pyinstaller_exe, spec_uninstall, "--noconfirm"], cwd=script_dir, check=True)
    
    dist_uninstall = os.path.join(script_dir, "dist", "Uninstall.exe")
    if os.path.exists(dist_uninstall):
        shutil.copy2(dist_uninstall, os.path.join(script_dir, "Uninstall.exe"))
        shutil.copy2(dist_uninstall, os.path.join(dist_app, "Uninstall.exe"))
        print("  [OK] Uninstall.exe bundled into AquaPulse_App")

    # Copy AquaPulse_App into C:\Users\parsa\Desktop\Install\windows\AquaPulse_App and script_dir
    dest_win_app = os.path.join(install_win_dir, "AquaPulse_App")
    force_remove_dir(dest_win_app)
    shutil.copytree(dist_app, dest_win_app)

    setup_app_src = os.path.join(script_dir, "AquaPulse_App")
    force_remove_dir(setup_app_src)
    shutil.copytree(dist_app, setup_app_src)

    # 3. Build AquaPulse_Setup.exe directly into C:\Users\parsa\Desktop\Install\windows
    print("\n[Step 3/4] Building multi-page Setup Wizard GUI (AquaPulse_Setup.exe)...")
    spec_setup = os.path.join(script_dir, "AquaPulse_Setup.spec")
    subprocess.run([pyinstaller_exe, spec_setup, "--noconfirm", "--distpath", install_win_dir], cwd=script_dir, check=True)

    setup_exe = os.path.join(install_win_dir, "AquaPulse_Setup.exe")
    
    # 4. Generate Docker Deployment Bundle in C:\Users\parsa\Desktop\Install\Docker
    print("\n[Step 4/4] Generating Docker Container Deployment Bundle...")
    create_docker_bundle(install_docker_dir, script_dir)

    readme_win_path = os.path.join(install_win_dir, "README_WINDOWS_INSTALLATION.txt")
    with open(readme_win_path, "w", encoding="utf-8") as f:
        f.write("""AquaPulse Windows Setup Suite
=======================================================
1. Run AquaPulse_Setup.exe to launch the multi-page GUI installer.
2. The setup wizard automatically audits system prerequisites (Ollama AI Engine, MiKTeX pdflatex) and configures Start Menu and Desktop shortcuts.
3. AquaPulse_App directory contains all standalone executable binaries, models, and assets.
""")

    if os.path.exists(setup_exe):
        print(f"\n=======================================================")
        print(f" SUCCESS! Build pipeline completed successfully!")
        print(f" Windows Target: {install_win_dir}")
        print(f"    • Setup Executable: {setup_exe}")
        print(f"    • App Package Dir:  {dest_win_app}")
        print(f" Docker Target:  {install_docker_dir}")
        print(f"    • Dockerfile & Compose Bundle: {install_docker_dir}")
        print(f"=======================================================")
    else:
        print(f"\n[ERROR] Build Failed: Could not produce {setup_exe}")

if __name__ == "__main__":
    main()
