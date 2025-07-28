#!/usr/bin/env python3
"""
Script de instalación completa para ROOP en Google Colab
Instala todas las dependencias necesarias
"""

import subprocess
import sys
import os

def install_package(package):
    """Instalar un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def install_requirements():
    """Instalar todas las dependencias necesarias para ROOP"""
    print("🚀 Instalando dependencias para ROOP...")
    
    # Lista de paquetes necesarios para ROOP
    packages = [
        "onnxruntime-gpu",  # Runtime de ONNX para GPU
        "tensorflow-gpu",   # TensorFlow para GPU
        "torch",            # PyTorch
        "torchvision",      # TorchVision
        "opencv-python",    # OpenCV
        "pillow",           # PIL/Pillow
        "numpy",            # NumPy
        "scipy",            # SciPy
        "scikit-image",     # Scikit-image
        "psutil",           # Para monitoreo de sistema
        "tqdm",             # Barras de progreso
        "insightface",      # Face recognition
        "basicsr",          # Super resolution
        "facexlib",         # Face detection
        "gfpgan",           # Face restoration
        "realesrgan",       # Real-ESRGAN
        "albumentations",   # Data augmentation
        "ffmpeg-python",    # FFmpeg wrapper
        "moviepy",          # Video processing
        "imageio",          # Image I/O
        "imageio-ffmpeg",   # FFmpeg support for imageio
    ]
    
    successful = 0
    failed = 0
    
    for package in packages:
        if install_package(package):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Resumen de instalación:")
    print(f"✅ Paquetes instalados: {successful}")
    print(f"❌ Paquetes fallidos: {failed}")
    
    return failed == 0

def setup_roop():
    """Configurar ROOP"""
    print("\n🔧 Configurando ROOP...")
    
    try:
        # Clonar ROOP si no existe
        if not os.path.exists("roop"):
            subprocess.run([
                "git", "clone", "--branch", "v3", "https://github.com/CDavidDv/roop.git"
            ], check=True)
            print("✅ ROOP clonado correctamente")
        else:
            print("✅ ROOP ya existe")
        
        # Cambiar al directorio de ROOP
        os.chdir("roop")
        
        # Descargar el modelo de face swap
        if not os.path.exists("inswapper_128.onnx"):
            subprocess.run([
                "wget", "https://civitai.com/api/download/models/85159", 
                "-O", "inswapper_128.onnx"
            ], check=True)
            print("✅ Modelo de face swap descargado")
        else:
            print("✅ Modelo de face swap ya existe")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configurando ROOP: {e}")
        return False

def verify_installation():
    """Verificar que todo esté instalado correctamente"""
    print("\n🔍 Verificando instalación...")
    
    try:
        # Verificar que onnxruntime esté disponible
        import onnxruntime
        print("✅ onnxruntime disponible")
        
        # Verificar que tensorflow esté disponible
        import tensorflow as tf
        print("✅ tensorflow disponible")
        
        # Verificar que torch esté disponible
        import torch
        print("✅ torch disponible")
        
        # Verificar que opencv esté disponible
        import cv2
        print("✅ opencv disponible")
        
        # Verificar que insightface esté disponible
        import insightface
        print("✅ insightface disponible")
        
        print("✅ Todas las dependencias están instaladas correctamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALADOR COMPLETO PARA ROOP EN GOOGLE COLAB")
    print("=" * 60)
    
    # Instalar dependencias
    if not install_requirements():
        print("❌ Error instalando dependencias")
        return
    
    # Configurar ROOP
    if not setup_roop():
        print("❌ Error configurando ROOP")
        return
    
    # Verificar instalación
    if not verify_installation():
        print("❌ Error en la verificación")
        return
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 60)
    print("✅ Todas las dependencias instaladas")
    print("✅ ROOP configurado")
    print("✅ Modelo de face swap descargado")
    print("\n🎬 Ahora puedes usar:")
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

if __name__ == "__main__":
    main() 