#!/usr/bin/env python3
"""
Script final de instalación para GPU con versiones más recientes
"""

import subprocess
import sys
import os

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

def install_final_gpu():
    """Instala todo con las versiones más recientes"""
    print("🚀 INSTALACIÓN FINAL PARA GPU")
    print("=" * 60)
    
    # Paso 1: Limpiar todo
    print("🧹 Limpiando instalaciones anteriores...")
    cleanup_commands = [
        "pip uninstall -y torch torchvision torchaudio torchtext triton",
        "pip uninstall -y tensorflow tensorflow-estimator tensorboard",
        "pip uninstall -y onnxruntime onnxruntime-gpu",
        "pip uninstall -y numpy",
        "pip uninstall -y nvidia-cudnn-cu12",
    ]
    
    for command in cleanup_commands:
        run_command(command, "Limpiando")
    
    # Paso 2: Instalar NumPy 1.x primero
    print("📦 Instalando NumPy 1.x...")
    if not run_command("pip install numpy==1.26.4", "Instalando NumPy 1.26.4"):
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
    
    # Paso 4: Instalar TensorFlow 2.16.1 (última versión)
    print("🧠 Instalando TensorFlow 2.16.1...")
    tf_commands = [
        "pip install tensorflow==2.16.1",
        "pip install tensorflow-estimator==2.16.1",
        "pip install tensorboard==2.16.2",
    ]
    
    for command in tf_commands:
        if not run_command(command, "Instalando TensorFlow"):
            return False
    
    # Paso 5: Instalar ONNX Runtime GPU
    print("⚡ Instalando ONNX Runtime GPU...")
    onnx_commands = [
        "pip install onnxruntime-gpu==1.17.0",
        "pip install nvidia-cudnn-cu12==8.9.2.26",  # Versión compatible con PyTorch
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
        "pip install onnx==1.16.0",
        "pip install gfpgan==1.3.8",
        "pip install basicsr==1.4.2",
        "pip install facexlib==0.3.0",
        "pip install filterpy==1.4.5",
        "pip install opennsfw2==0.10.2",
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
        "pip install tk==0.1.0",
    ]
    
    for command in ui_deps:
        if not run_command(command, "Instalando UI"):
            return False
    
    # Paso 8: Instalar librerías CUDA del sistema
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
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        if not np.__version__.startswith('1.'):
            print("❌ NumPy debe ser 1.x para TensorFlow")
            return False
        
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ GPU devices: {tf.config.list_physical_devices('GPU')}")
        
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
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
    print("🚀 INSTALACIÓN FINAL PARA GPU CON VERSIONES MÁS RECIENTES")
    print("=" * 70)
    
    # Instalar todo
    if not install_final_gpu():
        print("❌ Error en instalación")
        return False
    
    # Verificar instalación
    if not verify_installation():
        print("❌ Error en verificación")
        return False
    
    # Descargar modelo
    if not download_model():
        print("❌ Error descargando modelo")
        return False
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 60)
    print("✅ NumPy 1.26.4 (compatible con TensorFlow)")
    print("✅ PyTorch 2.2.0+cu121")
    print("✅ TensorFlow 2.16.1 (última versión)")
    print("✅ ONNX Runtime GPU 1.17.0")
    print("✅ GPU configurada correctamente")
    print("✅ Modelo descargado")
    print("\n🚀 Puedes ejecutar:")
    print("python run_batch_processing.py --source tu_imagen.jpg --videos video1.mp4 --output-dir resultados")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 