#!/usr/bin/env python3
"""
Script para descargar solo los modelos necesarios para ROOP
"""

import os
import sys
import subprocess
import shutil
import requests
from pathlib import Path

def clean_models():
    """Limpia modelos corruptos"""
    print("🧹 LIMPIANDO MODELOS CORRUPTOS")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    
    if os.path.exists(models_dir):
        print(f"🗑️ Eliminando directorio corrupto: {models_dir}")
        shutil.rmtree(models_dir)
    
    # Crear directorio limpio
    os.makedirs(models_dir, exist_ok=True)
    print("✅ Directorio de modelos limpio creado")

def download_inswapper():
    """Descarga el modelo inswapper_128.onnx"""
    print("📥 DESCARGANDO INSWAPPER_128.ONNX")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
    
    # URL del modelo inswapper_128.onnx
    inswapper_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
    
    try:
        print(f"🔄 Descargando inswapper_128.onnx desde: {inswapper_url}")
        response = requests.get(inswapper_url, stream=True)
        response.raise_for_status()
        
        with open(inswapper_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = os.path.getsize(inswapper_path)
        print(f"✅ inswapper_128.onnx descargado: {size:,} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando inswapper_128.onnx: {e}")
        return False

def download_detection_models():
    """Descarga modelos de detección desde buffalo_l.zip"""
    print("📥 DESCARGANDO MODELOS DE DETECCIÓN")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    
    try:
        # Descargar buffalo_l.zip
        zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        zip_path = "/tmp/buffalo_l.zip"
        
        print(f"🔄 Descargando buffalo_l.zip desde: {zip_url}")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ buffalo_l.zip descargado")
        
        # Extraer solo los modelos necesarios
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp/")
        
        # Mover solo los modelos de detección
        extracted_dir = "/tmp/buffalo_l"
        detection_models = ["det_10g.onnx", "2d106det.onnx"]
        
        for model in detection_models:
            src = os.path.join(extracted_dir, model)
            dst = os.path.join(models_dir, model)
            if os.path.exists(src):
                shutil.move(src, dst)
                size = os.path.getsize(dst)
                print(f"✅ {model} movido: {size:,} bytes")
            else:
                print(f"⚠️ {model} no encontrado en el ZIP")
        
        # Limpiar archivos temporales
        os.remove(zip_path)
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        
        print("✅ Modelos de detección descargados")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando modelos de detección: {e}")
        return False

def test_models():
    """Prueba que los modelos funcionen"""
    print("🧪 PROBANDO MODELOS")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    required_models = [
        "det_10g.onnx",
        "2d106det.onnx", 
        "inswapper_128.onnx"
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
    print("🚀 DESCARGANDO MODELOS CORRECTOS PARA ROOP")
    print("=" * 60)
    
    # Paso 1: Limpiar modelos corruptos
    clean_models()
    
    # Paso 2: Descargar inswapper_128.onnx
    if not download_inswapper():
        print("❌ Error descargando inswapper_128.onnx")
        return 1
    
    # Paso 3: Descargar modelos de detección
    if not download_detection_models():
        print("❌ Error descargando modelos de detección")
        return 1
    
    # Paso 4: Probar modelos
    if not test_models():
        print("❌ Los modelos no funcionan correctamente")
        return 1
    
    print("\n🎉 ¡MODELOS DESCARGADOS CORRECTAMENTE!")
    print("=" * 50)
    print("✅ inswapper_128.onnx - Modelo de face swap")
    print("✅ det_10g.onnx - Detección facial")
    print("✅ 2d106det.onnx - Landmarks 2D")
    print("✅ InsightFace funcionando")
    print("✅ Puedes procesar videos ahora")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 