#!/usr/bin/env python3
"""
Script completo para configurar ROOP con GPU en Google Colab
"""

import subprocess
import sys
import os
import shutil

def setup_colab_gpu():
    """Configurar ROOP con GPU en Google Colab"""
    print("🚀 CONFIGURANDO ROOP CON GPU EN GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar si estamos en Google Colab
    try:
        import google.colab
        print("✅ Detectado Google Colab")
    except ImportError:
        print("⚠️ No se detectó Google Colab, pero continuando...")
    
    # Paso 1: Instalar dependencias de GPU
    print("\n📦 PASO 1: Instalando dependencias de GPU...")
    
    # Desinstalar onnxruntime CPU si está instalado
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "onnxruntime", "-y"], 
                      capture_output=True)
        print("✅ onnxruntime CPU desinstalado")
    except:
        pass
    
    # Instalar onnxruntime-gpu
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.15.1"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ onnxruntime-gpu instalado correctamente")
        else:
            print(f"❌ Error instalando onnxruntime-gpu: {result.stderr}")
    except Exception as e:
        print(f"❌ Error con onnxruntime-gpu: {e}")
    
    # Instalar PyTorch con CUDA
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch==2.1.0+cu118", "torchvision==0.16.0+cu118", "torchaudio==2.1.0+cu118",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PyTorch con CUDA instalado correctamente")
        else:
            print(f"❌ Error instalando PyTorch: {result.stderr}")
    except Exception as e:
        print(f"❌ Error con PyTorch: {e}")
    
    # Instalar TensorFlow GPU
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ TensorFlow GPU instalado correctamente")
        else:
            print(f"❌ Error instalando TensorFlow: {result.stderr}")
    except Exception as e:
        print(f"❌ Error con TensorFlow: {e}")
    
    # Paso 2: Verificar instalación
    print("\n🔍 PASO 2: Verificando instalación...")
    
    # Verificar ONNX Runtime GPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU disponible")
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
    
    # Paso 3: Descargar modelos si no existen
    print("\n📥 PASO 3: Verificando modelos...")
    
    model_paths = [
        ("../inswapper_128.onnx", "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"),
        ("../models/GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth")
    ]
    
    for model_path, url in model_paths:
        if not os.path.exists(model_path):
            print(f"Descargando {model_path}...")
            try:
                import urllib.request
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ {model_path} descargado")
            except Exception as e:
                print(f"❌ Error descargando {model_path}: {e}")
        else:
            print(f"✅ {model_path} ya existe")
    
    # Paso 4: Probar face swapper con GPU
    print("\n🎭 PASO 4: Probando face swapper con GPU...")
    
    try:
        import roop.processors.frame.face_swapper as face_swapper
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            if hasattr(swapper, 'providers'):
                print(f"Proveedores: {swapper.providers}")
                if 'CUDAExecutionProvider' in swapper.providers:
                    print("🎉 ¡GPU funcionando en face swapper!")
                else:
                    print("⚠️ Face swapper usando CPU")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
    
    # Paso 5: Probar face enhancer con GPU
    print("\n✨ PASO 5: Probando face enhancer con GPU...")
    
    try:
        import roop.processors.frame.face_enhancer as face_enhancer
        device = face_enhancer.get_device()
        print(f"Dispositivo detectado: {device}")
        
        if device == 'cuda':
            print("✅ Face enhancer configurado para usar GPU")
        else:
            print("⚠️ Face enhancer usando CPU")
    except Exception as e:
        print(f"❌ Error probando face enhancer: {e}")
    
    print("\n🎉 CONFIGURACIÓN COMPLETADA!")
    print("=" * 60)
    print("💡 Ahora puedes ejecutar:")
    print("   python test_gpu_force.py")
    print("   python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4")
    print("\n💡 Para mejor rendimiento en Colab:")
    print("   python run_roop_with_memory_management.py --source imagen.jpg --target video.mp4 --output resultado.mp4 --gpu-memory-wait 30")

if __name__ == "__main__":
    setup_colab_gpu() 