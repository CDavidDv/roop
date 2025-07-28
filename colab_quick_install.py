#!/usr/bin/env python3
"""
Instalación rápida para ROOP en Google Colab
"""

import subprocess
import sys
import os

print("🚀 INSTALACIÓN RÁPIDA PARA ROOP EN GOOGLE COLAB")
print("=" * 50)

# Instalar dependencias esenciales
print("📦 Instalando dependencias...")

packages = [
    "onnxruntime-gpu",
    "tensorflow-gpu", 
    "torch",
    "torchvision",
    "opencv-python",
    "pillow",
    "numpy",
    "scipy",
    "psutil",
    "tqdm",
    "insightface",
    "basicsr",
    "facexlib",
    "gfpgan",
    "realesrgan",
    "albumentations",
    "ffmpeg-python",
    "moviepy",
    "imageio",
    "imageio-ffmpeg"
]

for package in packages:
    print(f"Instalando {package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ Dependencias instaladas")

# Clonar ROOP
print("\n🔧 Configurando ROOP...")
if not os.path.exists("roop"):
    subprocess.run(["git", "clone", "--branch", "v3", "https://github.com/CDavidDv/roop.git"])
    print("✅ ROOP clonado")
else:
    print("✅ ROOP ya existe")

# Cambiar al directorio
os.chdir("roop")

# Descargar modelo
if not os.path.exists("inswapper_128.onnx"):
    subprocess.run(["wget", "https://civitai.com/api/download/models/85159", "-O", "inswapper_128.onnx"])
    print("✅ Modelo descargado")
else:
    print("✅ Modelo ya existe")

print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
print("=" * 50)
print("Ahora puedes usar:")
print("!python run_batch_gpu_simple.py \\")
print("  --source /content/DanielaAS.jpg \\")
print("  --input-folder /content/videos \\")
print("  --output-folder /content/resultados \\")
print("  --frame-processors face_swapper face_enhancer \\")
print("  --max-memory 12 \\")
print("  --execution-threads 8 \\")
print("  --temp-frame-quality 100 \\")
print("  --gpu-memory-wait 30 \\")
print("  --keep-fps") 