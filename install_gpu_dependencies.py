#!/usr/bin/env python3
"""
Script para instalar dependencias de GPU en Google Colab
"""

import subprocess
import sys
import os

def install_gpu_dependencies():
    """Instalar dependencias de GPU para ROOP"""
    print("🚀 INSTALANDO DEPENDENCIAS DE GPU PARA ROOP")
    print("=" * 60)
    
    # Verificar si estamos en Google Colab
    try:
        import google.colab
        print("✅ Detectado Google Colab")
    except ImportError:
        print("⚠️ No se detectó Google Colab, pero continuando...")
    
    # Lista de paquetes a instalar
    packages = [
        "onnxruntime-gpu==1.15.1",
        "torch==2.1.0+cu118",
        "torchvision==0.16.0+cu118",
        "torchaudio==2.1.0+cu118",
        "tensorflow==2.15.0",
        "insightface==0.7.3",
        "opencv-python==4.8.0.74",
        "gfpgan==1.3.8",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "numpy==1.26.4",
        "tqdm==4.65.0"
    ]
    
    print("\n📦 Instalando paquetes de GPU...")
    
    for package in packages:
        try:
            print(f"Instalando {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                package, "--extra-index-url", "https://download.pytorch.org/whl/cu118"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} instalado correctamente")
            else:
                print(f"❌ Error instalando {package}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error con {package}: {e}")
    
    print("\n🔍 Verificando instalación...")
    
    # Verificar ONNX Runtime GPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU instalado correctamente")
        else:
            print("❌ ONNX Runtime GPU no disponible")
    except Exception as e:
        print(f"❌ Error verificando ONNX Runtime: {e}")
    
    # Verificar PyTorch CUDA
    try:
        import torch
        print(f"PyTorch CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU detectada: {torch.cuda.get_device_name()}")
            print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Error verificando PyTorch: {e}")
    
    # Verificar TensorFlow GPU
    try:
        import tensorflow as tf
        print(f"TensorFlow GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print(f"GPUs TensorFlow: {tf.config.list_physical_devices('GPU')}")
    except Exception as e:
        print(f"❌ Error verificando TensorFlow: {e}")
    
    print("\n🎉 Instalación completada!")
    print("=" * 60)
    print("💡 Ahora puedes ejecutar: python test_gpu_force.py")
    print("💡 Para procesar videos: python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4")

if __name__ == "__main__":
    install_gpu_dependencies() 