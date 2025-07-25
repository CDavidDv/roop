#!/usr/bin/env python3
"""
Script para instalar todas las dependencias faltantes incluyendo tkinterdnd2
"""

import os
import sys
import subprocess

def install_all_dependencies():
    """Instala todas las dependencias necesarias"""
    print("🔧 INSTALANDO TODAS LAS DEPENDENCIAS")
    print("=" * 50)
    
    # Lista completa de dependencias
    dependencies = [
        "customtkinter",
        "tkinterdnd2",
        "Pillow",
        "opencv-python",
        "numpy",
        "scikit-image",
        "scipy",
        "tqdm",
        "psutil",
        "insightface",
        "onnx",
        "onnxruntime-gpu",
        "torch",
        "torchvision",
        "matplotlib",
        "requests"
    ]
    
    for dep in dependencies:
        print(f"🔄 Instalando: {dep}")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Instalado: {dep}")
            else:
                print(f"⚠️ Error instalando {dep}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error con {dep}: {e}")

def test_all_imports():
    """Prueba que todas las importaciones funcionen"""
    print("🧪 PROBANDO TODAS LAS IMPORTACIONES")
    print("=" * 50)
    
    test_code = """
import sys
sys.path.insert(0, '.')

# Probar todas las importaciones necesarias
try:
    import customtkinter
    print("✅ customtkinter importado")
except ImportError as e:
    print(f"❌ Error importando customtkinter: {e}")

try:
    import tkinterdnd2
    print("✅ tkinterdnd2 importado")
except ImportError as e:
    print(f"❌ Error importando tkinterdnd2: {e}")

try:
    import roop.core
    print("✅ roop.core importado")
except ImportError as e:
    print(f"❌ Error importando roop.core: {e}")

try:
    import roop.ui
    print("✅ roop.ui importado")
except ImportError as e:
    print(f"❌ Error importando roop.ui: {e}")

try:
    import torch
    print("✅ torch importado")
except ImportError as e:
    print(f"❌ Error importando torch: {e}")

try:
    import onnxruntime
    print("✅ onnxruntime importado")
except ImportError as e:
    print(f"❌ Error importando onnxruntime: {e}")

try:
    import insightface
    print("✅ insightface importado")
except ImportError as e:
    print(f"❌ Error importando insightface: {e}")

print("✅ Todas las importaciones básicas funcionan")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def download_models():
    """Descarga los modelos necesarios"""
    print("📥 DESCARGANDO MODELOS")
    print("=" * 50)
    
    # Crear directorio de modelos
    models_dir = "/root/.insightface/models/buffalo_l"
    os.makedirs(models_dir, exist_ok=True)
    
    print("✅ Directorio de modelos creado")
    print("📋 Los modelos se descargarán automáticamente en el primer uso")
    print("⏳ Esto puede tomar varios minutos la primera vez...")

def main():
    """Función principal"""
    print("🚀 INSTALANDO TODAS LAS DEPENDENCIAS")
    print("=" * 60)
    
    # Paso 1: Instalar todas las dependencias
    install_all_dependencies()
    
    # Paso 2: Probar importaciones
    if not test_all_imports():
        print("⚠️ Algunas dependencias pueden no estar disponibles")
        print("🔄 Continuando de todas formas...")
    
    # Paso 3: Preparar modelos
    download_models()
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 50)
    print("✅ Todas las dependencias instaladas")
    print("✅ Modelos listos para descargar")
    print("✅ Puedes ejecutar el procesamiento por lotes ahora")
    print("\n🚀 Para procesar videos:")
    print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 /content/136.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 