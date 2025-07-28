#!/usr/bin/env python3
"""
Script para solucionar el problema de onnxruntime en Google Colab
"""

import subprocess
import sys
import os

def check_python_version():
    """Verificar versión de Python"""
    print(f"🐍 Python version: {sys.version}")
    print(f"🐍 Python executable: {sys.executable}")

def check_onnxruntime():
    """Verificar si onnxruntime está instalado"""
    try:
        import onnxruntime
        print(f"✅ onnxruntime disponible: {onnxruntime.__version__}")
        return True
    except ImportError:
        print("❌ onnxruntime NO está instalado")
        return False

def install_onnxruntime():
    """Instalar onnxruntime-gpu"""
    print("📦 Instalando onnxruntime-gpu...")
    
    try:
        # Intentar instalar onnxruntime-gpu
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu", "--force-reinstall"
        ], check=True)
        print("✅ onnxruntime-gpu instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando onnxruntime-gpu")
        return False

def install_alternative():
    """Instalar onnxruntime como alternativa"""
    print("📦 Intentando instalar onnxruntime...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime", "--force-reinstall"
        ], check=True)
        print("✅ onnxruntime instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando onnxruntime")
        return False

def check_cuda():
    """Verificar si CUDA está disponible"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("❌ CUDA no disponible")
            return False
    except ImportError:
        print("❌ PyTorch no disponible")
        return False

def install_all_dependencies():
    """Instalar todas las dependencias necesarias"""
    print("📦 Instalando todas las dependencias...")
    
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
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--force-reinstall"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package} instalado")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")

def test_imports():
    """Probar todas las importaciones necesarias"""
    print("\n🔍 Probando importaciones...")
    
    imports = [
        ("onnxruntime", "onnxruntime"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("insightface", "insightface"),
        ("basicsr", "basicsr"),
        ("facexlib", "facexlib"),
        ("gfpgan", "gfpgan"),
        ("realesrgan", "realesrgan")
    ]
    
    all_good = True
    for module_name, package_name in imports:
        try:
            __import__(module_name)
            print(f"✅ {package_name} importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando {package_name}: {e}")
            all_good = False
    
    return all_good

def main():
    """Función principal"""
    print("🔧 SOLUCIONADOR DE ONNXRUNTIME PARA GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar Python
    check_python_version()
    
    # Verificar CUDA
    check_cuda()
    
    # Verificar onnxruntime
    if not check_onnxruntime():
        print("\n🔄 Intentando solucionar onnxruntime...")
        
        # Intentar instalar onnxruntime-gpu
        if not install_onnxruntime():
            # Si falla, intentar con onnxruntime normal
            if not install_alternative():
                print("❌ No se pudo instalar onnxruntime")
                return
        
        # Verificar nuevamente
        if not check_onnxruntime():
            print("❌ onnxruntime aún no está disponible")
            return
    
    # Instalar todas las dependencias
    print("\n📦 Instalando todas las dependencias...")
    install_all_dependencies()
    
    # Probar importaciones
    if test_imports():
        print("\n🎉 ¡TODAS LAS DEPENDENCIAS ESTÁN LISTAS!")
        print("=" * 60)
        print("✅ onnxruntime instalado")
        print("✅ Todas las dependencias funcionando")
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
    else:
        print("\n❌ Algunas dependencias no están funcionando")
        print("🔄 Reinicia el runtime y ejecuta nuevamente")

if __name__ == "__main__":
    main() 