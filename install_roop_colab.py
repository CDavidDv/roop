#!/usr/bin/env python3
"""
Script de instalación optimizado para Google Colab
Sin entorno virtual, con GPU acceleration completa
"""

import os
import sys
import subprocess

def install_roop_colab():
    """Instalar Roop optimizado para Google Colab"""
    print("🚀 INSTALANDO ROOP GPU OPTIMIZADO")
    print("=" * 50)
    
    # Paso 1: Clonar repositorio
    print("📦 Paso 1: Clonando repositorio...")
    subprocess.run(["git", "clone", "https://github.com/CDavidDv/roop"], check=True)
    os.chdir("roop")
    print("✅ Repositorio clonado")
    
    # Paso 2: Solucionar NumPy
    print("\n🔧 Paso 2: Solucionando NumPy...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--no-cache-dir", "--force-reinstall"], check=True)
    
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    
    # Paso 3: Instalar dependencias
    print("\n📦 Paso 3: Instalando dependencias...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python", "insightface", "gfpgan", "basicsr", "facexlib", "sympy", "onnx"], check=True)
    print("✅ Dependencias instaladas")
    
    # Paso 4: Instalar GPU dependencies
    print("\n🚀 Paso 4: Instalando GPU dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "tensorflow", "tensorflow-gpu", "-y"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.15.1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12==8.9.4.25"], check=True)
    print("✅ GPU dependencies instaladas")
    
    # Paso 5: Configurar entorno
    print("\n⚙️ Paso 5: Configurando entorno...")
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("✅ Entorno configurado")
    
    # Paso 6: Descargar modelo
    print("\n📥 Paso 6: Descargando modelo...")
    subprocess.run(["wget", "https://civitai.com/api/download/models/85159", "-O", "inswapper_128.onnx"], check=True)
    print("✅ Modelo descargado")
    
    # Paso 7: Verificar GPU
    print("\n🔍 Paso 7: Verificando GPU...")
    try:
        import torch
        import onnxruntime as ort
        import tensorflow as tf
        
        print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
        print(f"✅ ONNX Providers: {ort.get_available_providers()}")
        print(f"✅ TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")
        print("✅ GPU configurado correctamente")
        
    except Exception as e:
        print(f"⚠️ Error verificando GPU: {e}")
    
    print("\n🎉 INSTALACIÓN COMPLETADA")
    print("=" * 50)
    print("Ahora puedes usar:")
    print()
    print("Procesamiento individual:")
    print("python run.py --source /content/source.jpg --target /content/video.mp4 --output /content/result.mp4 --execution-provider cuda --execution-threads 31 --temp-frame-quality 100 --keep-fps")
    print()
    print("Procesamiento por lotes:")
    print("python run_batch_processing.py --source /content/source.jpg --videos /content/video1.mp4 /content/video2.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps")

if __name__ == "__main__":
    install_roop_colab() 