#!/usr/bin/env python3
"""
Script para descargar correctamente los modelos de InsightFace
"""

import os
import sys
import subprocess
import shutil
import requests
from pathlib import Path

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

def download_models():
    """Descarga los modelos de InsightFace correctamente"""
    print("📥 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    # URLs de los modelos (versiones estables)
    model_urls = {
        "1k3d68.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "2d106det.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "det_10g.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "genderage.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "w600k_r50.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    }
    
    models_dir = "/root/.insightface/models/buffalo_l"
    
    try:
        # Descargar el archivo ZIP completo
        zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        zip_path = "/tmp/buffalo_l.zip"
        
        print(f"🔄 Descargando modelos desde: {zip_url}")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Archivo ZIP descargado")
        
        # Extraer el ZIP
        print("📦 Extrayendo modelos...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp/")
        
        # Mover archivos al directorio correcto
        extracted_dir = "/tmp/buffalo_l"
        if os.path.exists(extracted_dir):
            for file in os.listdir(extracted_dir):
                if file.endswith('.onnx'):
                    src = os.path.join(extracted_dir, file)
                    dst = os.path.join(models_dir, file)
                    shutil.move(src, dst)
                    print(f"✅ Movido: {file}")
        
        # Limpiar archivos temporales
        os.remove(zip_path)
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        
        print("✅ Modelos descargados y extraídos correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando modelos: {e}")
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
    return True
    
except Exception as e:
    print(f"❌ Error probando modelos: {e}")
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

def main():
    """Función principal"""
    print("🚀 DESCARGANDO MODELOS CORRECTAMENTE")
    print("=" * 60)
    
    # Paso 1: Limpiar modelos corruptos
    clean_models_directory()
    
    # Paso 2: Descargar modelos
    if not download_models():
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