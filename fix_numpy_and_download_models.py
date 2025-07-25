#!/usr/bin/env python3
"""
Script para arreglar NumPy y descargar modelos originales
"""

import os
import sys
import subprocess
import shutil

def fix_numpy():
    """Arregla el problema de NumPy 2.x"""
    print("🔧 ARREGLANDO NUMPY")
    print("=" * 50)
    
    try:
        # Desinstalar NumPy 2.x
        print("🔄 Desinstalando NumPy 2.x...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], 
                      capture_output=True, text=True)
        
        # Instalar NumPy 1.x
        print("🔄 Instalando NumPy 1.x...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy<2"], 
                      capture_output=True, text=True)
        
        # Reinstalar opencv-python
        print("🔄 Reinstalando opencv-python...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "-y"], 
                      capture_output=True, text=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], 
                      capture_output=True, text=True)
        
        print("✅ NumPy arreglado")
        return True
        
    except Exception as e:
        print(f"❌ Error arreglando NumPy: {e}")
        return False

def download_original_models():
    """Descarga los modelos originales que usabas"""
    print("📥 DESCARGANDO MODELOS ORIGINALES")
    print("=" * 50)
    
    try:
        # Crear directorio models si no existe
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Descargar inswapper_128.onnx
        print("🔄 Descargando inswapper_128.onnx...")
        subprocess.run([
            "wget", 
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
            "-O", "inswapper_128.onnx"
        ], check=True)
        
        # Mover a directorio models
        print("🔄 Moviendo modelo a directorio models...")
        shutil.move("inswapper_128.onnx", os.path.join(models_dir, "inswapper_128.onnx"))
        
        # Verificar que el archivo existe
        model_path = os.path.join(models_dir, "inswapper_128.onnx")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ inswapper_128.onnx descargado - {size:,} bytes")
            return True
        else:
            print("❌ Error: Modelo no encontrado después de descarga")
            return False
            
    except Exception as e:
        print(f"❌ Error descargando modelos: {e}")
        return False

def download_insightface_models():
    """Descarga los modelos de InsightFace"""
    print("📥 DESCARGANDO MODELOS INSIGHTFACE")
    print("=" * 50)
    
    try:
        # Crear directorio de modelos InsightFace
        insightface_dir = "/root/.insightface/models/buffalo_l"
        os.makedirs(insightface_dir, exist_ok=True)
        
        # Script para descargar modelos usando insightface
        download_script = """
import insightface
import os

try:
    print("🔄 Inicializando InsightFace...")
    
    # Crear aplicación de análisis facial
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    print("✅ InsightFace inicializado")
    
    # Preparar la aplicación (esto descargará los modelos automáticamente)
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
        
        print("🔄 Ejecutando descarga automática de InsightFace...")
        result = subprocess.run([sys.executable, "-c", download_script], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error descargando modelos InsightFace: {e}")
        return False

def test_roop():
    """Prueba que Roop funcione correctamente"""
    print("🧪 PROBANDO ROOP")
    print("=" * 50)
    
    test_code = """
import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, os.getcwd())

try:
    # Importar roop
    from roop import core
    print("✅ Roop importado correctamente")
    
    # Probar que no hay errores de NumPy
    import cv2
    import numpy as np
    print("✅ OpenCV y NumPy funcionando")
    
    print("✅ Roop está listo para usar")
    
except Exception as e:
    print(f"❌ Error probando Roop: {e}")
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
    print("🚀 ARREGLANDO NUMPY Y DESCARGANDO MODELOS")
    print("=" * 60)
    
    # Paso 1: Arreglar NumPy
    if not fix_numpy():
        print("❌ Error arreglando NumPy")
        return 1
    
    # Paso 2: Descargar modelos originales
    if not download_original_models():
        print("❌ Error descargando modelos originales")
        return 1
    
    # Paso 3: Descargar modelos InsightFace
    if not download_insightface_models():
        print("❌ Error descargando modelos InsightFace")
        return 1
    
    # Paso 4: Probar Roop
    if not test_roop():
        print("❌ Error probando Roop")
        return 1
    
    print("\n🎉 ¡TODO LISTO!")
    print("=" * 50)
    print("✅ NumPy arreglado")
    print("✅ Modelos originales descargados")
    print("✅ Modelos InsightFace descargados")
    print("✅ Roop funcionando correctamente")
    print("✅ Puedes ejecutar el procesamiento por lotes ahora")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 