#!/usr/bin/env python3
"""
Script de instalación rápida para ROOP en Google Colab
"""

import os
import subprocess
import sys
import time

def run_command(command, description):
    """Ejecutar comando con descripción"""
    print(f"\n🔄 {description}")
    print(f"💻 Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def install_dependencies():
    """Instalar todas las dependencias necesarias"""
    
    print("📦 INSTALANDO DEPENDENCIAS")
    print("=" * 60)
    
    dependencies = [
        "opencv-python",
        "pillow", 
        "onnxruntime-gpu",
        "opennsfw2",
        "insightface",
        "onnx",
        "tensorflow",
        "albumentations",
        "scikit-image",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "psutil",
        "GPUtil"
    ]
    
    successful = 0
    failed = 0
    
    for package in dependencies:
        try:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✅ {package} instalado")
            successful += 1
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")
            failed += 1
    
    print(f"\n📊 Resumen: {successful} exitosos, {failed} fallidos")
    return failed == 0

def setup_environment():
    """Configurar variables de entorno"""
    
    print("\n⚙️ CONFIGURANDO ENTORNO")
    print("=" * 60)
    
    # Configurar variables de entorno para GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("✅ Variables de entorno configuradas")
    
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

def create_folders():
    """Crear carpetas necesarias"""
    
    print("\n📁 CREANDO CARPETAS")
    print("=" * 60)
    
    folders = [
        "/content/videos",
        "/content/resultados", 
        "/content/sources"
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Carpeta creada: {folder}")
        else:
            print(f"📁 Carpeta ya existe: {folder}")

def download_model():
    """Descargar modelo de face swap"""
    
    print("\n📥 DESCARGANDO MODELO")
    print("=" * 60)
    
    model_path = "inswapper_128.onnx"
    
    if os.path.exists(model_path):
        print(f"✅ Modelo ya existe: {model_path}")
        return True
    
    try:
        print("📥 Descargando modelo de face swap...")
        subprocess.check_call([
            "wget", 
            "https://civitai.com/api/download/models/85159", 
            "-O", model_path,
            "--quiet"
        ])
        print("✅ Modelo descargado exitosamente")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error descargando modelo")
        return False

def verify_installation():
    """Verificar que todo esté instalado correctamente"""
    
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 60)
    
    critical_packages = [
        "cv2",  # opencv-python
        "PIL",  # pillow
        "onnxruntime",  # onnxruntime-gpu
        "opennsfw2",  # opennsfw2
        "insightface",  # insightface
        "torch",  # torch
        "numpy"  # numpy
    ]
    
    all_ok = True
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO INSTALADO")
            all_ok = False
    
    return all_ok

def show_next_steps():
    """Mostrar próximos pasos"""
    
    print("\n🎯 PRÓXIMOS PASOS")
    print("=" * 60)
    print("1. 📸 Sube tu imagen fuente a: /content/sources/")
    print("2. 🎬 Sube tus videos a: /content/videos/")
    print("3. 🚀 Ejecuta el procesamiento:")
    print("   python run_colab_gpu.py")
    print("4. 📁 Los resultados estarán en: /content/resultados/")
    print("=" * 60)

def main():
    """Función principal"""
    
    print("🚀 INSTALACIÓN RÁPIDA PARA ROOP GPU")
    print("=" * 60)
    
    start_time = time.time()
    
    # Instalar dependencias
    deps_ok = install_dependencies()
    
    # Configurar entorno
    setup_environment()
    
    # Crear carpetas
    create_folders()
    
    # Descargar modelo
    model_ok = download_model()
    
    # Verificar instalación
    verify_ok = verify_installation()
    
    # Mostrar próximos pasos
    show_next_steps()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo total: {elapsed_time:.1f} segundos")
    
    if deps_ok and model_ok and verify_ok:
        print("\n✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
        print("🚀 ¡Listo para procesar videos con GPU!")
    else:
        print("\n⚠️ INSTALACIÓN INCOMPLETA")
        print("💡 Revisa los errores arriba")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 