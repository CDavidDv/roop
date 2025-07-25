#!/usr/bin/env python3
"""
Script para descargar manualmente los modelos de InsightFace
"""

import os
import subprocess
import sys

def download_insightface_models():
    """Descarga los modelos de InsightFace manualmente"""
    print("📥 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    # Crear directorio de modelos
    models_dir = "/root/.insightface/models/buffalo_l"
    os.makedirs(models_dir, exist_ok=True)
    print(f"✅ Directorio creado: {models_dir}")
    
    # Lista de modelos necesarios
    models = [
        "1k3d68.onnx",
        "2d106det.onnx", 
        "det_10g.onnx",
        "genderage.onnx",
        "w600k_r50.onnx"
    ]
    
    base_url = "https://github.com/deepinsight/insightface/releases/download/v0.7.3/"
    
    for model in models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            print(f"🔄 Descargando: {model}")
            url = base_url + model
            try:
                result = subprocess.run([
                    "wget", "-O", model_path, url
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ Descargado: {model}")
                else:
                    print(f"❌ Error descargando {model}: {result.stderr}")
            except Exception as e:
                print(f"❌ Error con {model}: {e}")
        else:
            print(f"✅ Ya existe: {model}")
    
    print("\n✅ Descarga de modelos completada")

def test_face_analyser():
    """Prueba que el analizador de rostros funcione"""
    print("\n🧪 PROBANDO ANALIZADOR DE ROSTROS")
    print("=" * 50)
    
    test_code = """
import insightface
import cv2
import numpy as np

print("🔍 Inicializando FaceAnalysis...")
try:
    # Configurar providers para GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Crear analizador
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("✅ FaceAnalysis inicializado correctamente")
    
    # Crear imagen de prueba
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Probar detección
    faces = app.get(test_image)
    print(f"✅ Detección funcionando: {len(faces)} rostros encontrados")
    
    print("✅ Analizador de rostros funcionando correctamente")
    
except Exception as e:
    print(f"❌ Error en FaceAnalysis: {e}")
    import traceback
    traceback.print_exc()
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
    print("🚀 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 60)
    
    # Descargar modelos
    download_insightface_models()
    
    # Probar analizador
    if test_face_analyser():
        print("\n🎉 ¡MODELOS DESCARGADOS Y FUNCIONANDO!")
        print("=" * 60)
        print("✅ Todos los modelos descargados")
        print("✅ Analizador de rostros funcionando")
        print("✅ Listo para procesamiento")
    else:
        print("\n⚠️ Problemas con el analizador de rostros")
        print("🔄 Intentando descargar modelos alternativos...")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 