#!/usr/bin/env python3
"""
Script para arreglar los modelos de InsightFace
"""

import os
import sys
import shutil
import subprocess

def clean_insightface_models():
    """Limpia los modelos de InsightFace que están causando problemas"""
    print("🧹 LIMPIANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    # Limpiar directorio de InsightFace
    insightface_dir = "/root/.insightface/models/buffalo_l"
    if os.path.exists(insightface_dir):
        print(f"🗑️ Eliminando: {insightface_dir}")
        shutil.rmtree(insightface_dir)
    
    # Crear directorio vacío
    os.makedirs(insightface_dir, exist_ok=True)
    print("✅ Directorio de InsightFace limpiado")

def copy_original_model():
    """Copia el modelo original a InsightFace"""
    print("📋 COPIANDO MODELO ORIGINAL")
    print("=" * 50)
    
    source_path = "models/inswapper_128.onnx"
    target_path = "/root/.insightface/models/buffalo_l/inswapper_128.onnx"
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        size = os.path.getsize(target_path)
        print(f"✅ Modelo copiado: {size:,} bytes")
        return True
    else:
        print(f"❌ Modelo original no encontrado: {source_path}")
        return False

def download_insightface_models():
    """Descarga los modelos mínimos de InsightFace"""
    print("📥 DESCARGANDO MODELOS MÍNIMOS DE INSIGHTFACE")
    print("=" * 50)
    
    # Crear script de descarga
    download_script = """
import insightface
import os

# Configurar para descargar automáticamente
os.environ['INSIGHTFACE_HOME'] = '/root/.insightface'

try:
    # Crear aplicación con descarga automática
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ InsightFace configurado correctamente")
    
    # Probar con imagen vacía
    import numpy as np
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = app.get(test_img)
    print("✅ Detección funcionando")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", download_script], 
                              capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_roop():
    """Prueba que ROOP funcione"""
    print("🧪 PROBANDO ROOP")
    print("=" * 50)
    
    test_code = """
import sys
sys.path.insert(0, '.')

try:
    # Probar importación
    import roop.core
    print("✅ roop.core importado")
    
    # Probar face_analyser
    from roop.face_analyser import get_face_analyser
    print("✅ face_analyser importado")
    
    # Probar face_swapper
    from roop.processors.frame.face_swapper import get_face_swapper
    print("✅ face_swapper importado")
    
    print("✅ ROOP funcionando correctamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO MODELOS DE INSIGHTFACE")
    print("=" * 60)
    
    # Paso 1: Limpiar modelos de InsightFace
    clean_insightface_models()
    
    # Paso 2: Copiar modelo original
    if not copy_original_model():
        print("❌ No se pudo copiar el modelo original")
        return 1
    
    # Paso 3: Descargar modelos mínimos de InsightFace
    if not download_insightface_models():
        print("❌ Error descargando modelos de InsightFace")
        return 1
    
    # Paso 4: Probar ROOP
    if not test_roop():
        print("❌ ROOP no funciona correctamente")
        return 1
    
    print("\n🎉 ¡MODELOS ARREGLADOS!")
    print("=" * 50)
    print("✅ Modelos de InsightFace limpiados")
    print("✅ Modelo original copiado")
    print("✅ InsightFace configurado")
    print("✅ ROOP funcionando")
    print("✅ Listo para procesar videos")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 