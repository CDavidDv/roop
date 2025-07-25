#!/usr/bin/env python3
"""
Script de instalación optimizado para GPU con manejo robusto de dependencias
"""

import subprocess
import sys
import os
import platform

def run_command(command, description=""):
    """Ejecuta un comando y maneja errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Exitoso")
            return True
        else:
            print(f"❌ {description} - Error")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Excepción: {e}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("✅ Versión de Python compatible")
        return True
    else:
        print("❌ Se requiere Python 3.10+")
        return False

def install_system_dependencies():
    """Instala dependencias del sistema"""
    commands = [
        ("apt-get update", "Actualizando repositorios"),
        ("apt-get install -y ffmpeg", "Instalando ffmpeg"),
        ("pip install --upgrade pip", "Actualizando pip"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def install_python_dependencies():
    """Instala dependencias de Python con manejo robusto"""
    
    # Paso 1: Limpiar instalaciones anteriores
    print("🧹 Limpiando instalaciones anteriores...")
    cleanup_commands = [
        "pip uninstall -y torch torchvision torchaudio torchtext triton",
        "pip uninstall -y tensorflow tensorflow-estimator tensorboard",
        "pip uninstall -y onnxruntime onnxruntime-gpu",
        "pip uninstall -y numpy",
    ]
    
    for command in cleanup_commands:
        run_command(command, "Limpiando")
    
    # Paso 2: Instalar NumPy primero
    print("📦 Instalando NumPy...")
    numpy_versions = ["2.2.0", "2.1.4", "2.1.0", "1.26.4"]
    
    numpy_installed = False
    for version in numpy_versions:
        if run_command(f"pip install numpy=={version}", f"Instalando NumPy {version}"):
            numpy_installed = True
            break
    
    if not numpy_installed:
        print("❌ No se pudo instalar NumPy")
        return False
    
    # Paso 3: Instalar PyTorch con CUDA
    print("🔥 Instalando PyTorch con CUDA...")
    torch_commands = [
        "pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121",
        "pip install torchtext==0.17.0 triton==2.2.0",
    ]
    
    for command in torch_commands:
        if not run_command(command, "Instalando PyTorch"):
            return False
    
    # Paso 4: Instalar TensorFlow con versión compatible
    print("🧠 Instalando TensorFlow...")
    tf_commands = [
        "pip install tensorflow==2.15.0",
        "pip install tensorflow-estimator==2.15.0",
        "pip install tensorboard==2.15.0",
    ]
    
    for command in tf_commands:
        if not run_command(command, "Instalando TensorFlow"):
            return False
    
    # Paso 5: Instalar ONNX Runtime GPU
    print("⚡ Instalando ONNX Runtime GPU...")
    onnx_commands = [
        "pip install onnxruntime-gpu==1.17.0",
        "pip install nvidia-cudnn-cu12==8.9.4.25",
    ]
    
    for command in onnx_commands:
        if not run_command(command, "Instalando ONNX Runtime"):
            return False
    
    # Paso 6: Instalar otras dependencias
    print("📚 Instalando otras dependencias...")
    other_deps = [
        "pip install opencv-python==4.9.0.80",
        "pip install insightface==0.7.3",
        "pip install face-alignment==1.3.5",
        "pip install psutil==5.9.8",
        "pip install tqdm==4.66.1",
        "pip install pillow==10.2.0",
        "pip install typing-extensions==4.10.0",
    ]
    
    for command in other_deps:
        if not run_command(command, "Instalando dependencias"):
            return False
    
    # Paso 7: Instalar dependencias de UI
    print("🖥️ Instalando dependencias de UI...")
    ui_deps = [
        "pip install customtkinter==5.2.2",
        "pip install darkdetect==0.8.0",
        "pip install tkinterdnd2==0.3.0",
    ]
    
    for command in ui_deps:
        if not run_command(command, "Instalando UI"):
            return False
    
    return True

def install_cuda_libraries():
    """Instala librerías CUDA del sistema"""
    print("🔧 Instalando librerías CUDA del sistema...")
    
    cuda_commands = [
        "apt-get install -y libcublas-11-8",
        "apt-get install -y libcublas-dev-11-8", 
        "apt-get install -y libcudnn8",
        "apt-get install -y libcudnn8-dev",
        "apt-get install -y libnvinfer8",
        "apt-get install -y libnvinfer-dev",
    ]
    
    for command in cuda_commands:
        if not run_command(command, "Instalando librerías CUDA"):
            return False
    
    return True

def verify_installation():
    """Verifica que todo esté instalado correctamente"""
    print("🔍 Verificando instalación...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ GPU devices: {tf.config.list_physical_devices('GPU')}")
        
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import customtkinter
        print(f"✅ CustomTkinter: {customtkinter.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

def download_model():
    """Descarga el modelo de face swap"""
    print("📥 Descargando modelo de face swap...")
    
    model_url = "https://civitai.com/api/download/models/85159"
    model_path = "inswapper_128.onnx"
    
    if not os.path.exists(model_path):
        if run_command(f"wget {model_url} -O {model_path}", "Descargando modelo"):
            print("✅ Modelo descargado")
        else:
            print("❌ Error descargando modelo")
            return False
    else:
        print("✅ Modelo ya existe")
    
    return True

def main():
    """Función principal"""
    print("🚀 INICIANDO INSTALACIÓN OPTIMIZADA PARA GPU")
    print("=" * 60)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar dependencias del sistema
    if not install_system_dependencies():
        return False
    
    # Instalar dependencias de Python
    if not install_python_dependencies():
        return False
    
    # Instalar librerías CUDA
    if not install_cuda_libraries():
        return False
    
    # Verificar instalación
    if not verify_installation():
        return False
    
    # Descargar modelo
    if not download_model():
        return False
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 60)
    print("✅ Todas las dependencias instaladas")
    print("✅ GPU configurada correctamente")
    print("✅ Modelo descargado")
    print("\n🚀 Puedes ejecutar:")
    print("python run_batch_processing.py --source tu_imagen.jpg --videos video1.mp4 --output-dir resultados")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 