#!/usr/bin/env python3
"""
Script para descargar modelos directamente usando insightface
"""

import os
import sys
import subprocess
import shutil

def clean_models_directory():
    """Limpia el directorio de modelos corruptos"""
    print("🧹 LIMPIANDO MODELOS CORRUPTOS")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    
    if os.path.exists(models_dir):
        print(f"🗑️ Eliminando directorio corrupto: {models_dir}")
        shutil.rmtree(models_dir)
    
    # Crear directorio limpio
    os.makedirs(models_dir, exist_ok=True)
    print("✅ Directorio de modelos limpio creado")

def download_models_direct():
    """Descarga los modelos directamente usando insightface"""
    print("📥 DESCARGANDO MODELOS DIRECTAMENTE")
    print("=" * 50)
    
    # Script para descargar modelos usando insightface
    download_script = """
import insightface
import os

try:
    print("🔄 Inicializando InsightFace...")
    
    # Crear aplicación de análisis facial (esto descargará automáticamente los modelos)
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    print("✅ InsightFace inicializado")
    
    # Preparar la aplicación (esto descargará los modelos si no existen)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Modelos descargados automáticamente")
    
    # Verificar que los modelos existan
    models_dir = "/root/.insightface/models/buffalo_l"
    required_models = [
        "1k3d68.onnx",
        "2d106det.onnx", 
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx"
    ]
    
    print("📋 Verificando modelos descargados:")
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"  ✅ {model} - {size:,} bytes")
        else:
            print(f"  ❌ {model} - NO ENCONTRADO")
    
    print("✅ Descarga de modelos completada")
    
except Exception as e:
    print(f"❌ Error descargando modelos: {e}")
    raise
"""
    
    try:
        print("🔄 Ejecutando descarga automática...")
        result = subprocess.run([sys.executable, "-c", download_script], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en descarga: {e}")
        return False

def test_models():
    """Prueba que los modelos funcionen correctamente"""
    print("🧪 PROBANDO MODELOS")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    required_models = [
        "1k3d68.onnx",
        "2d106det.onnx", 
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx"
    ]
    
    # Verificar que todos los modelos existan
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ {model} - {size:,} bytes")
        else:
            print(f"❌ {model} - NO ENCONTRADO")
            return False
    
    # Probar InsightFace
    test_code = """
import insightface
import cv2
import numpy as np

try:
    # Crear aplicación de análisis facial
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ InsightFace inicializado correctamente")
    
    # Crear imagen de prueba
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Probar detección
    faces = app.get(test_img)
    print("✅ Detección facial funcionando")
    
    print("✅ Todos los modelos funcionan correctamente")
    
except Exception as e:
    print(f"❌ Error probando modelos: {e}")
    raise
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

def main():
    """Función principal"""
    print("🚀 DESCARGANDO MODELOS DIRECTAMENTE")
    print("=" * 60)
    
    # Paso 1: Limpiar modelos corruptos
    clean_models_directory()
    
    # Paso 2: Descargar modelos usando insightface
    if not download_models_direct():
        print("❌ Error descargando modelos")
        return 1
    
    # Paso 3: Probar modelos
    if not test_models():
        print("❌ Los modelos no funcionan correctamente")
        return 1
    
    print("\n🎉 ¡MODELOS DESCARGADOS CORRECTAMENTE!")
    print("=" * 50)
    print("✅ Modelos limpios descargados")
    print("✅ InsightFace funcionando")
    print("✅ Puedes ejecutar el procesamiento por lotes ahora")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 