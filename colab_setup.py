#!/usr/bin/env python3
"""
Script de configuración para Google Colab - ROOP GPU Processing
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_colab_environment():
    """Configurar el entorno de Colab para ROOP"""
    
    print("🚀 CONFIGURANDO ENTORNO PARA GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar si estamos en Colab
    try:
        import google.colab
        print("✅ Detectado Google Colab")
    except ImportError:
        print("⚠️ No se detectó Google Colab, pero continuando...")
    
    # Configurar variables de entorno para GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("✅ Variables de entorno configuradas para GPU")
    
    # Verificar GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"✅ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ No se detectó GPU CUDA")
    except ImportError:
        print("⚠️ PyTorch no disponible para verificar GPU")
    
    print("=" * 60)

def create_folders():
    """Crear carpetas necesarias"""
    
    folders = [
        "/content/videos",
        "/content/resultados",
        "/content/sources"
    ]
    
    print("📁 Creando carpetas necesarias...")
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Carpeta creada: {folder}")
        else:
            print(f"📁 Carpeta ya existe: {folder}")
    
    print("=" * 60)

def download_model():
    """Descargar el modelo de face swap si no existe"""
    
    model_path = "inswapper_128.onnx"
    
    if os.path.exists(model_path):
        print(f"✅ Modelo ya existe: {model_path}")
        return
    
    print("📥 Descargando modelo de face swap...")
    
    try:
        # Intentar descargar desde CivitAI
        cmd = [
            "wget", 
            "https://civitai.com/api/download/models/85159", 
            "-O", model_path
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Modelo descargado exitosamente")
        
    except subprocess.CalledProcessError:
        print("❌ Error descargando modelo desde CivitAI")
        print("💡 Puedes descargar manualmente el modelo 'inswapper_128.onnx'")
        return False
    
    return True

def check_dependencies():
    """Verificar dependencias necesarias"""
    
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "opencv-python",
        "numpy",
        "pillow",
        "onnxruntime-gpu"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - FALTANTE")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Paquetes faltantes: {', '.join(missing_packages)}")
        print("💡 Instala con: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ Todas las dependencias están instaladas")
        return True

def show_usage_instructions():
    """Mostrar instrucciones de uso"""
    
    print("\n📋 INSTRUCCIONES DE USO")
    print("=" * 60)
    print("1. 📸 Sube tu imagen fuente a: /content/sources/")
    print("2. 🎬 Sube tus videos a: /content/videos/")
    print("3. 🚀 Ejecuta el procesamiento con:")
    print("   python run_colab_gpu.py")
    print("4. 📁 Los resultados estarán en: /content/resultados/")
    print("=" * 60)
    print("\n💡 CONFIGURACIÓN PERSONALIZADA:")
    print("Edita las variables en run_colab_gpu.py:")
    print("- SOURCE_PATH: Ruta de tu imagen fuente")
    print("- INPUT_FOLDER: Carpeta con videos")
    print("- OUTPUT_FOLDER: Carpeta para resultados")
    print("=" * 60)

def main():
    """Función principal de configuración"""
    
    print("🎯 CONFIGURACIÓN AUTOMÁTICA PARA ROOP GPU")
    print("=" * 60)
    
    # Configurar entorno
    setup_colab_environment()
    
    # Crear carpetas
    create_folders()
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    # Descargar modelo
    model_ok = download_model()
    
    # Mostrar instrucciones
    show_usage_instructions()
    
    if deps_ok and model_ok:
        print("\n✅ CONFIGURACIÓN COMPLETADA")
        print("🚀 Listo para procesar videos con GPU")
    else:
        print("\n⚠️ CONFIGURACIÓN INCOMPLETA")
        print("💡 Revisa los errores arriba antes de continuar")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 