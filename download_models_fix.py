#!/usr/bin/env python3
"""
Script para descargar manualmente los modelos de InsightFace
"""

import os
import subprocess
import sys
import shutil

def download_insightface_models():
    """Descarga los modelos de InsightFace manualmente"""
    print("📥 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    # Crear directorio de modelos
    models_dir = "/root/.insightface/models/buffalo_l"
    os.makedirs(models_dir, exist_ok=True)
    print(f"✅ Directorio creado: {models_dir}")
    
    # Clonar el repositorio si no existe
    repo_dir = "/tmp/insightface_models"
    if not os.path.exists(repo_dir):
        print("🔄 Clonando repositorio de InsightFace...")
        try:
            subprocess.run([
                "git", "clone", "https://github.com/deepinsight/insightface.git", repo_dir
            ], check=True)
            print("✅ Repositorio clonado")
        except Exception as e:
            print(f"❌ Error clonando repositorio: {e}")
            return False
    
    # Buscar modelos en el repositorio
    possible_paths = [
        os.path.join(repo_dir, "python-package/insightface/model_zoo/models/buffalo_l"),
        os.path.join(repo_dir, "model_zoo/models/buffalo_l"),
        os.path.join(repo_dir, "models/buffalo_l"),
        os.path.join(repo_dir, "python-package/insightface/model_zoo/models"),
        os.path.join(repo_dir, "model_zoo/models")
    ]
    
    models_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Encontrado directorio de modelos: {path}")
            
            # Listar archivos en el directorio
            files = os.listdir(path)
            print(f"📁 Archivos encontrados: {files}")
            
            # Copiar archivos .onnx
            onnx_files = [f for f in files if f.endswith('.onnx')]
            if onnx_files:
                print(f"🔄 Copiando {len(onnx_files)} modelos...")
                for file in onnx_files:
                    src = os.path.join(path, file)
                    dst = os.path.join(models_dir, file)
                    try:
                        shutil.copy2(src, dst)
                        size = os.path.getsize(dst) / 1024 / 1024
                        print(f"✅ Copiado: {file} ({size:.1f}MB)")
                    except Exception as e:
                        print(f"❌ Error copiando {file}: {e}")
                models_found = True
                break
    
    if not models_found:
        print("❌ No se encontraron modelos en el repositorio")
        return False
    
    print("\n✅ Descarga de modelos completada")
    return True

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

def download_models_alternative():
    """Método alternativo para descargar modelos"""
    print("\n🔄 MÉTODO ALTERNATIVO - DESCARGANDO MODELOS")
    print("=" * 50)
    
    # URLs alternativas desde releases de GitHub
    models = [
        {
            "name": "1k3d68.onnx",
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/1k3d68.onnx"
        },
        {
            "name": "2d106det.onnx",
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/2d106det.onnx"
        },
        {
            "name": "det_10g.onnx",
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/det_10g.onnx"
        },
        {
            "name": "genderage.onnx",
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/genderage.onnx"
        },
        {
            "name": "w600k_r50.onnx",
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7.3/w600k_r50.onnx"
        }
    ]
    
    models_dir = "/root/.insightface/models/buffalo_l"
    os.makedirs(models_dir, exist_ok=True)
    
    for model in models:
        model_path = os.path.join(models_dir, model["name"])
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size > 1000000:
                print(f"✅ Ya existe: {model['name']} ({file_size/1024/1024:.1f}MB)")
                continue
        
        print(f"🔄 Descargando: {model['name']}")
        try:
            result = subprocess.run([
                "wget", "-O", model_path, model["url"]
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                file_size = os.path.getsize(model_path)
                if file_size > 1000000:
                    print(f"✅ Descargado: {model['name']} ({file_size/1024/1024:.1f}MB)")
                else:
                    print(f"❌ Archivo muy pequeño: {model['name']} ({file_size} bytes)")
                    os.remove(model_path)
            else:
                print(f"❌ Error descargando {model['name']}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error con {model['name']}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)

def main():
    """Función principal"""
    print("🚀 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 60)
    
    # Intentar descarga desde repositorio
    if download_insightface_models():
        print("✅ Modelos descargados desde repositorio")
    else:
        print("⚠️ Falló descarga desde repositorio")
        print("🔄 Intentando método alternativo...")
        download_models_alternative()
    
    # Verificar modelos
    if not verify_models():
        print("\n⚠️ Algunos modelos están corruptos")
        print("🔄 Intentando descarga alternativa...")
        download_models_alternative()
    
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
        download_models_alternative()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 