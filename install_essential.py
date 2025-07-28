#!/usr/bin/env python3
"""
Script de instalación de dependencias esenciales para ROOP (sin GUI)
"""

import subprocess
import sys
import os
import time

def install_package(package_name, description=""):
    """Instalar un paquete con descripción"""
    try:
        print(f"📦 Instalando {package_name}...")
        if description:
            print(f"   {description}")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
        print(f"✅ {package_name} instalado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package_name}: {e}")
        return False

def install_essential_dependencies():
    """Instalar solo las dependencias esenciales (sin GUI)"""
    
    print("🚀 INSTALACIÓN DE DEPENDENCIAS ESENCIALES")
    print("=" * 60)
    
    # Solo dependencias esenciales (sin GUI)
    dependencies = [
        ("opencv-python", "Procesamiento de imágenes y videos"),
        ("pillow", "Procesamiento de imágenes"),
        ("onnxruntime-gpu", "Runtime de ONNX para GPU"),
        ("opennsfw2", "Detección de contenido NSFW"),
        ("insightface", "Reconocimiento facial"),
        ("onnx", "Formato de modelo ONNX"),
        ("tensorflow", "Framework de machine learning"),
        ("albumentations", "Aumentación de datos"),
        ("scikit-image", "Procesamiento de imágenes"),
        ("scipy", "Cálculos científicos"),
        ("numpy", "Cálculos numéricos"),
        ("torch", "PyTorch para deep learning"),
        ("torchvision", "Vision models para PyTorch"),
        ("tqdm", "Barras de progreso"),
        ("psutil", "Información del sistema"),
        ("GPUtil", "Información de GPU")
    ]
    
    successful = 0
    failed = 0
    
    for package, description in dependencies:
        if install_package(package, description):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 RESUMEN: {successful} exitosos, {failed} fallidos")
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
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("onnxruntime", "ONNX Runtime"),
        ("opennsfw2", "OpenNSFW2"),
        ("insightface", "InsightFace"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow")
    ]
    
    all_ok = True
    
    for package, name in critical_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NO INSTALADO")
            all_ok = False
    
    return all_ok

def show_usage_examples():
    """Mostrar ejemplos de uso"""
    
    print("\n🎯 EJEMPLOS DE USO")
    print("=" * 60)
    print("1. Procesamiento básico (modo headless):")
    print("   python run_headless.py")
    print()
    print("2. Con imagen personalizada:")
    print("   python run_headless.py --source /content/sources/mi_imagen.jpg")
    print()
    print("3. Con carpetas personalizadas:")
    print("   python run_headless.py --source /content/sources/mi_imagen.jpg \\")
    print("   --input-folder /content/mis_videos --output-folder /content/mis_resultados")
    print()
    print("4. Con parámetros optimizados:")
    print("   python run_headless.py --max-memory 10 --execution-threads 20")
    print("=" * 60)

def main():
    """Función principal"""
    
    print("🎯 INSTALACIÓN ESENCIAL PARA ROOP GPU (SIN GUI)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Instalar dependencias esenciales
    deps_ok = install_essential_dependencies()
    
    # Configurar entorno
    setup_environment()
    
    # Crear carpetas
    create_folders()
    
    # Descargar modelo
    model_ok = download_model()
    
    # Verificar instalación
    verify_ok = verify_installation()
    
    # Mostrar ejemplos de uso
    show_usage_examples()
    
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo total: {elapsed_time:.1f} segundos")
    
    if deps_ok and model_ok and verify_ok:
        print("\n✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
        print("🚀 ¡Listo para procesar videos con GPU!")
        print("\n💡 Próximos pasos:")
        print("1. Sube tu imagen fuente a /content/sources/")
        print("2. Sube tus videos a /content/videos/")
        print("3. Ejecuta: python run_headless.py")
    else:
        print("\n⚠️ INSTALACIÓN INCOMPLETA")
        print("💡 Revisa los errores arriba")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 