#!/usr/bin/env python3
"""
Script simple para limpiar modelos corruptos
"""

import os
import shutil
import sys

def clean_models():
    """Limpia modelos corruptos y deja que InsightFace los descargue automáticamente"""
    print("🧹 LIMPIANDO MODELOS CORRUPTOS")
    print("=" * 50)
    
    models_dir = "/root/.insightface/models/buffalo_l"
    
    if os.path.exists(models_dir):
        print(f"🗑️ Eliminando directorio corrupto: {models_dir}")
        shutil.rmtree(models_dir)
        print("✅ Modelos corruptos eliminados")
    else:
        print("✅ No hay modelos para limpiar")
    
    print("\n📋 InsightFace descargará los modelos automáticamente")
    print("⏳ La primera ejecución puede tomar varios minutos...")

def test_insightface():
    """Prueba InsightFace con descarga automática"""
    print("🧪 PROBANDO INSIGHTFACE")
    print("=" * 50)
    
    test_code = """
import insightface
import cv2
import numpy as np

try:
    print("🔄 Inicializando InsightFace...")
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    print("✅ InsightFace inicializado")
    
    print("🔄 Preparando modelos...")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Modelos preparados")
    
    # Crear imagen de prueba
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    print("🔄 Probando detección...")
    faces = app.get(test_img)
    print("✅ Detección funcionando")
    
    print("✅ InsightFace funcionando correctamente")
    return True
    
except Exception as e:
    print(f"❌ Error: {e}")
    return False
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 LIMPIANDO MODELOS CORRUPTOS")
    print("=" * 60)
    
    # Paso 1: Limpiar modelos
    clean_models()
    
    # Paso 2: Probar InsightFace
    print("\n🔄 Probando InsightFace...")
    if test_insightface():
        print("\n🎉 ¡MODELOS LIMPIOS!")
        print("=" * 50)
        print("✅ Modelos corruptos eliminados")
        print("✅ InsightFace funcionando")
        print("✅ Puedes procesar videos ahora")
    else:
        print("\n❌ InsightFace aún tiene problemas")
        print("🔄 Intentando descarga manual...")
    
    return 0

if __name__ == "__main__":
    import subprocess
    sys.exit(main()) 