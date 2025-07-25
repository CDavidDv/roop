#!/usr/bin/env python3
"""
Script para descargar manualmente los modelos de InsightFace
"""

import os
import subprocess
import sys
import requests

def download_insightface_models():
    """Descarga los modelos de InsightFace manualmente"""
    print("📥 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    # Crear directorio de modelos
    models_dir = "/root/.insightface/models/buffalo_l"
    os.makedirs(models_dir, exist_ok=True)
    print(f"✅ Directorio creado: {models_dir}")
    
    # Lista de modelos con URLs alternativas
    models = [
        {
            "name": "1k3d68.onnx",
            "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/1k3d68.onnx"
        },
        {
            "name": "2d106det.onnx",
            "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/2d106det.onnx"
        },
        {
            "name": "det_10g.onnx",
            "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx"
        },
        {
            "name": "genderage.onnx",
            "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/genderage.onnx"
        },
        {
            "name": "w600k_r50.onnx",
            "url": "https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
        }
    ]
    
    for model in models:
        model_path = os.path.join(models_dir, model["name"])
        
        # Verificar si el archivo existe y tiene tamaño correcto
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > 1000000:  # Más de 1MB
                print(f"✅ Ya existe: {model['name']} ({file_size/1024/1024:.1f}MB)")
                continue
            else:
                print(f"⚠️ Archivo corrupto: {model['name']} ({file_size} bytes)")
                os.remove(model_path)
        
        print(f"🔄 Descargando: {model['name']}")
        
        try:
            # Usar requests para mejor control
            response = requests.get(model["url"], stream=True, timeout=300)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verificar tamaño del archivo descargado
            file_size = os.path.getsize(model_path)
            if file_size > 1000000:  # Más de 1MB
                print(f"✅ Descargado: {model['name']} ({file_size/1024/1024:.1f}MB)")
            else:
                print(f"❌ Archivo muy pequeño: {model['name']} ({file_size} bytes)")
                os.remove(model_path)
                
        except Exception as e:
            print(f"❌ Error descargando {model['name']}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
    
    print("\n✅ Descarga de modelos completada")

def verify_models():
    """Verifica que los modelos estén correctos"""
    print("\n🔍 VERIFICANDO MODELOS")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    required_models = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"]
    
    all_good = True
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > 1000000:
                print(f"✅ {model}: {file_size/1024/1024:.1f}MB")
            else:
                print(f"❌ {model}: Archivo muy pequeño ({file_size} bytes)")
                all_good = False
        else:
            print(f"❌ {model}: No encontrado")
            all_good = False
    
    return all_good

def test_face_analyser():
    """Prueba que el analizador de rostros funcione"""
    print("\n🧪 PROBANDO ANALIZADOR DE ROSTROS")
    print("=" * 50)
    
    test_code = """
import insightface
import cv2
import numpy as np
import os

print("🔍 Verificando modelos...")
models_dir = "/root/.insightface/models/buffalo_l"
required_models = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"]

for model in required_models:
    model_path = os.path.join(models_dir, model)
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024 / 1024
        print(f"✅ {model}: {size:.1f}MB")
    else:
        print(f"❌ {model}: No encontrado")

print("\\n🔍 Inicializando FaceAnalysis...")
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

def force_download_models():
    """Fuerza la descarga de modelos usando git lfs"""
    print("\n🔄 DESCARGANDO MODELOS CON GIT LFS")
    print("=" * 50)
    
    try:
        # Clonar el repositorio de modelos
        models_repo = "/tmp/insightface_models"
        if os.path.exists(models_repo):
            subprocess.run(["rm", "-rf", models_repo], check=True)
        
        print("🔄 Clonando repositorio de modelos...")
        subprocess.run([
            "git", "clone", "https://github.com/deepinsight/insightface.git", models_repo
        ], check=True)
        
        # Copiar modelos
        source_dir = os.path.join(models_repo, "python-package/insightface/model_zoo/models/buffalo_l")
        target_dir = "/root/.insightface/models/buffalo_l"
        
        if os.path.exists(source_dir):
            print("🔄 Copiando modelos...")
            subprocess.run(["cp", "-r", f"{source_dir}/*", target_dir], check=True)
            print("✅ Modelos copiados")
        else:
            print("❌ Directorio de modelos no encontrado")
            
    except Exception as e:
        print(f"❌ Error con git lfs: {e}")

def main():
    """Función principal"""
    print("🚀 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 60)
    
    # Intentar descarga normal
    download_insightface_models()
    
    # Verificar modelos
    if not verify_models():
        print("\n⚠️ Algunos modelos están corruptos")
        print("🔄 Intentando descarga alternativa...")
        force_download_models()
        download_insightface_models()  # Intentar de nuevo
    
    # Probar analizador
    if test_face_analyser():
        print("\n🎉 ¡MODELOS DESCARGADOS Y FUNCIONANDO!")
        print("=" * 60)
        print("✅ Todos los modelos descargados")
        print("✅ Analizador de rostros funcionando")
        print("✅ Listo para procesamiento")
        return True
    else:
        print("\n❌ Problemas con el analizador de rostros")
        print("🔄 Intentando descarga manual...")
        force_download_models()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 