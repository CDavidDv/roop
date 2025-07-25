#!/usr/bin/env python3
"""
Script de instalación rápida para ROOP
Maneja versiones compatibles automáticamente
"""

import os
import sys
import subprocess
import platform

def run_command(command, description=""):
    """Ejecutar comando con manejo de errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 10):
        print("❌ Se requiere Python 3.10+")
        return False
    else:
        print("✅ Versión de Python compatible")
        return True

def install_numpy_compatible():
    """Instalar NumPy con versión compatible"""
    print("\n📦 Instalando NumPy compatible...")
    
    # Desinstalar NumPy anterior
    run_command("pip uninstall -y numpy", "Desinstalando NumPy anterior")
    
    # Intentar diferentes versiones de NumPy
    numpy_versions = ["2.2.0", "2.1.4", "2.1.3", "2.1.2", "2.1.1", "2.1.0", "2.0.2", "2.0.1", "2.0.0"]
    
    for version in numpy_versions:
        print(f"🔧 Intentando NumPy {version}...")
        if run_command(f"pip install numpy=={version}", f"Instalando NumPy {version}"):
            print(f"✅ NumPy {version} instalado exitosamente")
            return True
    
    # Si ninguna versión específica funciona, instalar la más reciente disponible
    print("🔧 Instalando versión más reciente de NumPy...")
    return run_command("pip install numpy", "Instalando NumPy más reciente")

def install_core_dependencies():
    """Instalar dependencias principales"""
    print("\n🐍 Instalando dependencias principales...")
    
    # Desinstalar versiones conflictivas
    run_command("pip uninstall -y torch torchvision torchaudio", "Desinstalando PyTorch anterior")
    run_command("pip uninstall -y tensorflow", "Desinstalando TensorFlow anterior")
    run_command("pip uninstall -y onnxruntime onnxruntime-gpu", "Desinstalando ONNX Runtime anterior")
    
    # Instalar PyTorch con CUDA 12.1
    run_command("pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121", 
                "Instalando PyTorch 2.2.0 con CUDA 12.1")
    
    # Instalar TensorFlow 2.16.1
    run_command("pip install tensorflow==2.16.1", "Instalando TensorFlow 2.16.1")
    
    # Instalar ONNX Runtime GPU
    run_command("pip install onnxruntime-gpu==1.17.0", "Instalando ONNX Runtime GPU 1.17.0")
    
    return True

def install_vision_dependencies():
    """Instalar dependencias de visión"""
    print("\n👁️ Instalando dependencias de visión...")
    
    vision_deps = [
        "opencv-python==4.9.0.80",
        "insightface==0.7.3",
        "gfpgan==1.3.8",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "filterpy==1.4.5",
        "opennsfw2==0.10.2"
    ]
    
    for dep in vision_deps:
        run_command(f"pip install {dep}", f"Instalando {dep}")
    
    return True

def install_ui_dependencies():
    """Instalar dependencias de UI"""
    print("\n🖥️ Instalando dependencias de UI...")
    
    ui_deps = [
        "customtkinter==5.2.2",
        "darkdetect==0.8.0",
        "tkinterdnd2==0.3.0",
        "tk==0.1.0"
    ]
    
    for dep in ui_deps:
        run_command(f"pip install {dep}", f"Instalando {dep}")
    
    return True

def install_other_dependencies():
    """Instalar otras dependencias"""
    print("\n📚 Instalando otras dependencias...")
    
    other_deps = [
        "pillow==10.2.0",
        "tqdm==4.66.1",
        "psutil==5.9.8",
        "coloredlogs==15.0.1",
        "humanfriendly==10.0",
        "sqlalchemy==2.0.31",
        "addict==2.4.0",
        "pydantic==2.8.0",
        "pydantic-core==2.20.0",
        "lmdb==1.5.1",
        "typing-extensions==4.10.0"
    ]
    
    for dep in other_deps:
        run_command(f"pip install {dep}", f"Instalando {dep}")
    
    return True

def download_model():
    """Descargar modelo de face swap"""
    print("\n📥 Descargando modelo de face swap...")
    
    model_url = "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"
    model_path = "inswapper_128.onnx"
    
    if not os.path.exists(model_path):
        run_command(f"wget {model_url} -O {model_path}", "Descargando modelo")
    else:
        print("✅ Modelo ya existe")
    
    return True

def verify_installation():
    """Verificar que la instalación fue exitosa"""
    print("\n🔍 Verificando instalación...")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPUs TensorFlow: {len(gpus)}")
        
        import onnxruntime as ort
        print(f"✅ ONNX Runtime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"✅ Proveedores ONNX: {providers}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import insightface
        print(f"✅ InsightFace: {insightface.__version__}")
        
        try:
            import customtkinter as ctk
            print(f"✅ CustomTkinter: {ctk.__version__}")
        except ImportError:
            print("⚠️ CustomTkinter no disponible (opcional)")
        
        print("✅ Todas las dependencias principales instaladas correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error verificando instalación: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("🚀 INICIANDO INSTALACIÓN RÁPIDA DE ROOP")
    print("=" * 60)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar NumPy compatible
    if not install_numpy_compatible():
        print("❌ Error instalando NumPy")
        return False
    
    # Instalar dependencias principales
    if not install_core_dependencies():
        print("❌ Error instalando dependencias principales")
        return False
    
    # Instalar dependencias de visión
    if not install_vision_dependencies():
        print("❌ Error instalando dependencias de visión")
        return False
    
    # Instalar dependencias de UI
    if not install_ui_dependencies():
        print("⚠️ Error instalando dependencias de UI (continuando...)")
    
    # Instalar otras dependencias
    if not install_other_dependencies():
        print("❌ Error instalando otras dependencias")
        return False
    
    # Verificar instalación
    if not verify_installation():
        print("❌ Error verificando instalación")
        return False
    
    # Descargar modelo
    if not download_model():
        print("❌ Error descargando modelo")
        return False
    
    print("\n" + "=" * 60)
    print("✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 Próximos pasos:")
    print("1. Ejecuta: python run.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("2. Para procesamiento en lote: python run_batch_processing.py --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados")
    print("3. El sistema está optimizado para GPU")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main() 