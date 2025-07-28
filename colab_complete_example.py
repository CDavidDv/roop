#!/usr/bin/env python3
"""
Ejemplo completo para Google Colab
Incluye instalación y procesamiento de videos
"""

import subprocess
import sys
import os

def install_dependencies():
    """Instalar dependencias necesarias"""
    print("🚀 Instalando dependencias...")
    
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

def setup_roop():
    """Configurar ROOP"""
    print("\n🔧 Configurando ROOP...")
    
    # Clonar ROOP
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

def process_videos():
    """Procesar videos"""
    print("\n🎬 Procesando videos...")
    
    # Comando para procesar videos
    cmd = [
        "python", "run_batch_gpu_simple.py",
        "--source", "/content/DanielaAS.jpg",  # Cambiar por tu imagen
        "--input-folder", "/content/videos",   # Carpeta con videos
        "--output-folder", "/content/resultados",  # Carpeta de salida
        "--frame-processors", "face_swapper", "face_enhancer",
        "--max-memory", "12",
        "--execution-threads", "8",
        "--temp-frame-quality", "100",
        "--gpu-memory-wait", "30",
        "--keep-fps"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Procesamiento completado exitosamente!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en el procesamiento: {e}")

def main():
    """Función principal"""
    print("🎬 EJEMPLO COMPLETO PARA GOOGLE COLAB")
    print("=" * 50)
    
    # Paso 1: Instalar dependencias
    install_dependencies()
    
    # Paso 2: Configurar ROOP
    setup_roop()
    
    # Paso 3: Procesar videos
    process_videos()
    
    print("\n🎉 ¡PROCESAMIENTO COMPLETADO!")
    print("=" * 50)
    print("📁 Videos guardados en: /content/resultados")

if __name__ == "__main__":
    main() 