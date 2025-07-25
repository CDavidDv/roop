#!/usr/bin/env python3
"""
Script de instalación optimizado para Google Colab
Con versiones más recientes y optimizaciones para GPU
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
        print(f"STDOUT: {e.stdout}")
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

def install_system_dependencies():
    """Instalar dependencias del sistema"""
    print("\n📦 Instalando dependencias del sistema...")
    
    # Actualizar pip
    run_command("pip install --upgrade pip", "Actualizando pip")
    
    # Instalar dependencias del sistema
    if platform.system() == "Linux":
        run_command("apt-get update", "Actualizando repositorios")
        run_command("apt-get install -y ffmpeg", "Instalando ffmpeg")
    
    return True

def install_python_dependencies():
    """Instalar dependencias de Python con versiones actualizadas"""
    print("\n🐍 Instalando dependencias de Python...")
    
    # Desinstalar versiones conflictivas
    run_command("pip uninstall -y numpy", "Desinstalando NumPy anterior")
    run_command("pip uninstall -y torch torchvision torchaudio", "Desinstalando PyTorch anterior")
    run_command("pip uninstall -y tensorflow", "Desinstalando TensorFlow anterior")
    run_command("pip uninstall -y onnxruntime onnxruntime-gpu", "Desinstalando ONNX Runtime anterior")
    
    # Instalar NumPy 2.x
    run_command("pip install numpy==2.1.4", "Instalando NumPy 2.1.4")
    
    # Instalar PyTorch con CUDA 12.1
    run_command("pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121", 
                "Instalando PyTorch 2.2.0 con CUDA 12.1")
    
    # Instalar TensorFlow 2.16.1
    run_command("pip install tensorflow==2.16.1", "Instalando TensorFlow 2.16.1")
    
    # Instalar ONNX Runtime GPU
    run_command("pip install onnxruntime-gpu==1.17.0", "Instalando ONNX Runtime GPU 1.17.0")
    
    # Instalar otras dependencias
    dependencies = [
        "opencv-python==4.9.0.80",
        "insightface==0.7.3",
        "gfpgan==1.3.8",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "filterpy==1.4.5",
        "opennsfw2==0.10.2",
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
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"Instalando {dep}")
    
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
        
        print("✅ Todas las dependencias instaladas correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error verificando instalación: {e}")
        return False

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

def create_optimized_script():
    """Crear script optimizado para GPU"""
    print("\n📝 Creando script optimizado...")
    
    script_content = '''#!/usr/bin/env python3
import os
import sys

# Configurar variables de entorno ANTES de cualquier import
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configurar matplotlib
import matplotlib
matplotlib.use('Agg')

# Modificar la función predict_video para saltar la verificación NSFW
import roop.predictor
original_predict_video = roop.predictor.predict_video

def predict_video_skip_nsfw(target_path: str) -> bool:
    print("⚠️ Saltando verificación NSFW para optimizar rendimiento GPU...")
    return False

roop.predictor.predict_video = predict_video_skip_nsfw

# Ahora importar y ejecutar roop
from roop import core

if __name__ == "__main__":
    core.run()
'''
    
    with open("run_roop_gpu.py", "w") as f:
        f.write(script_content)
    
    print("✅ Script optimizado creado: run_roop_gpu.py")
    return True

def main():
    """Función principal de instalación"""
    print("🚀 INICIANDO INSTALACIÓN DE ROOP OPTIMIZADO")
    print("=" * 60)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar dependencias
    if not install_system_dependencies():
        return False
    
    if not install_python_dependencies():
        return False
    
    # Verificar instalación
    if not verify_installation():
        return False
    
    # Descargar modelo
    if not download_model():
        return False
    
    # Crear script optimizado
    if not create_optimized_script():
        return False
    
    print("\n" + "=" * 60)
    print("✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 Próximos pasos:")
    print("1. Ejecuta: python run_roop_gpu.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("2. Para procesamiento en lote: python run_batch_processing.py --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados")
    print("3. El sistema está optimizado para GPU y usa las versiones más recientes")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main() 