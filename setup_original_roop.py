#!/usr/bin/env python3
"""
Script para configurar ROOP como funcionaba originalmente
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def setup_original_structure():
    """Configura la estructura original de ROOP"""
    print("🏗️ CONFIGURANDO ESTRUCTURA ORIGINAL")
    print("=" * 50)
    
    # Crear directorio models
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    print(f"✅ Directorio creado: {models_dir}")
    
    return models_dir

def download_original_inswapper():
    """Descarga inswapper_128.onnx de HuggingFace (método original)"""
    print("📥 DESCARGANDO INSWAPPER_128.ONNX (MÉTODO ORIGINAL)")
    print("=" * 50)
    
    models_dir = "models"
    inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
    
    # URL original de HuggingFace
    inswapper_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    
    try:
        print(f"🔄 Descargando desde: {inswapper_url}")
        response = requests.get(inswapper_url, stream=True)
        response.raise_for_status()
        
        with open(inswapper_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = os.path.getsize(inswapper_path)
        print(f"✅ inswapper_128.onnx descargado: {size:,} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return False

def check_original_model():
    """Verifica que el modelo esté en el lugar correcto"""
    print("🔍 VERIFICANDO MODELO ORIGINAL")
    print("=" * 50)
    
    models_dir = "models"
    inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
    
    if os.path.exists(inswapper_path):
        size = os.path.getsize(inswapper_path)
        print(f"✅ inswapper_128.onnx encontrado: {size:,} bytes")
        print(f"📁 Ubicación: {os.path.abspath(inswapper_path)}")
        return True
    else:
        print(f"❌ inswapper_128.onnx no encontrado en: {inswapper_path}")
        return False

def test_original_roop():
    """Prueba que ROOP funcione con el modelo original"""
    print("🧪 PROBANDO ROOP ORIGINAL")
    print("=" * 50)
    
    test_code = """
import sys
sys.path.insert(0, '.')

try:
    # Verificar que el modelo existe
    import os
    model_path = "models/inswapper_128.onnx"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Modelo encontrado: {size:,} bytes")
    else:
        print("❌ Modelo no encontrado")
        exit(1)
    
    # Probar importación de ROOP
    import roop.core
    print("✅ roop.core importado")
    
    # Probar face_swapper
    from roop.processors.frame.face_swapper import get_face_swapper
    print("✅ face_swapper importado")
    
    print("✅ ROOP original funcionando correctamente")
    return True
    
except Exception as e:
    print(f"❌ Error: {e}")
    return False
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

def show_usage():
    """Muestra cómo usar ROOP original"""
    print("📋 USO DE ROOP ORIGINAL")
    print("=" * 50)
    print("🚀 Para procesar videos:")
    print("   python run.py --target /content/1.mp4 --source /content/AriaAS.jpg -o /content/swapped.mp4 --execution-provider cuda --frame-processor face_swapper face_enhancer")
    print()
    print("🚀 Para procesamiento por lotes:")
    print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
    print()
    print("📁 Estructura de archivos:")
    print("   roop/")
    print("   ├── models/")
    print("   │   └── inswapper_128.onnx")
    print("   ├── run.py")
    print("   └── run_batch_processing.py")

def main():
    """Función principal"""
    print("🚀 CONFIGURANDO ROOP ORIGINAL")
    print("=" * 60)
    
    # Paso 1: Configurar estructura
    models_dir = setup_original_structure()
    
    # Paso 2: Descargar modelo original
    if not download_original_inswapper():
        print("❌ Error descargando modelo original")
        return 1
    
    # Paso 3: Verificar modelo
    if not check_original_model():
        print("❌ Modelo no encontrado")
        return 1
    
    # Paso 4: Probar ROOP
    if not test_original_roop():
        print("❌ ROOP no funciona correctamente")
        return 1
    
    print("\n🎉 ¡ROOP ORIGINAL CONFIGURADO!")
    print("=" * 50)
    print("✅ Estructura original creada")
    print("✅ inswapper_128.onnx descargado")
    print("✅ ROOP funcionando")
    print("✅ Listo para procesar videos")
    
    show_usage()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 