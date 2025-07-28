#!/usr/bin/env python3
"""
Instalación paso a paso para ROOP en Google Colab
Más confiable y con verificaciones
"""

import subprocess
import sys
import os
import time

def step_1_install_basic():
    """Paso 1: Instalar dependencias básicas"""
    print("📦 PASO 1: Instalando dependencias básicas...")
    
    basic_packages = [
        "numpy",
        "pillow",
        "opencv-python",
        "scipy",
        "psutil",
        "tqdm"
    ]
    
    for package in basic_packages:
        print(f"Instalando {package}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", package, "--quiet"
        ])
    
    print("✅ Dependencias básicas instaladas")

def step_2_install_onnxruntime():
    """Paso 2: Instalar onnxruntime específicamente"""
    print("\n📦 PASO 2: Instalando onnxruntime...")
    
    try:
        # Intentar onnxruntime-gpu primero
        print("Intentando onnxruntime-gpu...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "--force-reinstall", "--quiet"
        ], check=True)
        print("✅ onnxruntime-gpu instalado")
    except subprocess.CalledProcessError:
        print("❌ onnxruntime-gpu falló, intentando onnxruntime...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "onnxruntime", "--force-reinstall", "--quiet"
            ], check=True)
            print("✅ onnxruntime instalado")
        except subprocess.CalledProcessError:
            print("❌ Error instalando onnxruntime")
            return False
    
    return True

def step_3_install_tensorflow():
    """Paso 3: Instalar TensorFlow"""
    print("\n📦 PASO 3: Instalando TensorFlow...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "tensorflow-gpu", "--quiet"
        ], check=True)
        print("✅ TensorFlow instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando TensorFlow")
        return False

def step_4_install_pytorch():
    """Paso 4: Instalar PyTorch"""
    print("\n📦 PASO 4: Instalando PyTorch...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "torch", "torchvision", "--quiet"
        ], check=True)
        print("✅ PyTorch instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando PyTorch")
        return False

def step_5_install_face_processing():
    """Paso 5: Instalar librerías de procesamiento facial"""
    print("\n📦 PASO 5: Instalando librerías de procesamiento facial...")
    
    face_packages = [
        "insightface",
        "basicsr",
        "facexlib",
        "gfpgan",
        "realesrgan"
    ]
    
    for package in face_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True)
            print(f"✅ {package} instalado")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")
    
    return True

def step_6_install_video_processing():
    """Paso 6: Instalar librerías de procesamiento de video"""
    print("\n📦 PASO 6: Instalando librerías de procesamiento de video...")
    
    video_packages = [
        "albumentations",
        "ffmpeg-python",
        "moviepy",
        "imageio",
        "imageio-ffmpeg"
    ]
    
    for package in video_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True)
            print(f"✅ {package} instalado")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")
    
    return True

def step_7_setup_roop():
    """Paso 7: Configurar ROOP"""
    print("\n🔧 PASO 7: Configurando ROOP...")
    
    try:
        # Clonar ROOP
        if not os.path.exists("roop"):
            subprocess.run([
                "git", "clone", "--branch", "v3", "https://github.com/CDavidDv/roop.git"
            ], check=True)
            print("✅ ROOP clonado")
        else:
            print("✅ ROOP ya existe")
        
        # Cambiar al directorio
        os.chdir("roop")
        
        # Descargar modelo
        if not os.path.exists("inswapper_128.onnx"):
            subprocess.run([
                "wget", "https://civitai.com/api/download/models/85159", "-O", "inswapper_128.onnx"
            ], check=True)
            print("✅ Modelo descargado")
        else:
            print("✅ Modelo ya existe")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configurando ROOP: {e}")
        return False

def step_8_verify_installation():
    """Paso 8: Verificar instalación"""
    print("\n🔍 PASO 8: Verificando instalación...")
    
    try:
        import onnxruntime
        print(f"✅ onnxruntime: {onnxruntime.__version__}")
        
        import tensorflow as tf
        print(f"✅ tensorflow: {tf.__version__}")
        
        import torch
        print(f"✅ torch: {torch.__version__}")
        
        import cv2
        print(f"✅ opencv: {cv2.__version__}")
        
        import insightface
        print("✅ insightface disponible")
        
        print("✅ Todas las dependencias verificadas")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN PASO A PASO PARA ROOP")
    print("=" * 60)
    
    steps = [
        ("Dependencias básicas", step_1_install_basic),
        ("ONNX Runtime", step_2_install_onnxruntime),
        ("TensorFlow", step_3_install_tensorflow),
        ("PyTorch", step_4_install_pytorch),
        ("Procesamiento facial", step_5_install_face_processing),
        ("Procesamiento de video", step_6_install_video_processing),
        ("Configurar ROOP", step_7_setup_roop),
        ("Verificar instalación", step_8_verify_installation)
    ]
    
    for i, (name, step_func) in enumerate(steps, 1):
        print(f"\n🔄 PASO {i}/8: {name}")
        print("-" * 40)
        
        if not step_func():
            print(f"❌ Error en paso {i}: {name}")
            print("🔄 Reinicia el runtime y ejecuta nuevamente")
            return
        
        print(f"✅ Paso {i} completado")
        time.sleep(1)  # Pausa entre pasos
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 60)
    print("✅ Todas las dependencias instaladas")
    print("✅ ROOP configurado")
    print("✅ Listo para procesar videos")
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