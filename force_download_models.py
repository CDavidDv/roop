#!/usr/bin/env python3
"""
Script simple para forzar la descarga de modelos de InsightFace
"""

import os
import sys
import subprocess

def force_insightface_download():
    """Fuerza la descarga de modelos usando InsightFace directamente"""
    print("🚀 FORZANDO DESCARGA DE MODELOS INSIGHTFACE")
    print("=" * 60)
    
    # Script que inicializa InsightFace y fuerza la descarga
    download_script = '''
import insightface
import cv2
import numpy as np
import os

print("🔍 Inicializando InsightFace...")

try:
    # Configurar providers para GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    print("📥 Descargando modelos (esto puede tomar varios minutos)...")
    
    # Crear analizador - esto descargará automáticamente los modelos
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    
    print("✅ Analizador creado, preparando...")
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("✅ FaceAnalysis inicializado correctamente")
    
    # Crear imagen de prueba
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Probar detección
    faces = app.get(test_image)
    print(f"✅ Detección funcionando: {len(faces)} rostros encontrados")
    
    print("✅ Todos los modelos descargados y funcionando")
    
    # Verificar que los archivos existen
    models_dir = "/root/.insightface/models/buffalo_l"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        print(f"📁 Modelos descargados: {files}")
        for file in files:
            if file.endswith('.onnx'):
                size = os.path.getsize(os.path.join(models_dir, file)) / 1024 / 1024
                print(f"  ✅ {file}: {size:.1f}MB")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        print("🔄 Ejecutando descarga automática...")
        result = subprocess.run([sys.executable, "-c", download_script], 
                              capture_output=True, text=True, timeout=600)  # 10 minutos timeout
        
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        
        if result.returncode == 0:
            print("\n🎉 ¡MODELOS DESCARGADOS EXITOSAMENTE!")
            return True
        else:
            print("\n❌ Error en la descarga")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Timeout - la descarga tomó demasiado tiempo")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_models():
    """Prueba que los modelos funcionen"""
    print("\n🧪 PROBANDO MODELOS DESCARGADOS")
    print("=" * 50)
    
    test_script = '''
import insightface
import cv2
import numpy as np

print("🔍 Probando modelos descargados...")

try:
    # Configurar providers para GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Crear analizador
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("✅ FaceAnalysis funcionando")
    
    # Crear imagen de prueba
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Probar detección
    faces = app.get(test_image)
    print(f"✅ Detección funcionando: {len(faces)} rostros encontrados")
    
    print("✅ Todos los modelos funcionan correctamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
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
    print("🚀 SOLUCIÓN DIRECTA PARA MODELOS INSIGHTFACE")
    print("=" * 60)
    print("📋 Este script fuerza la descarga automática de modelos")
    print("⚡ Usa el método oficial de InsightFace")
    print("=" * 60)
    
    # Forzar descarga
    if force_insightface_download():
        print("\n✅ Descarga completada")
        
        # Probar modelos
        if test_models():
            print("\n🎉 ¡TODO FUNCIONANDO!")
            print("=" * 60)
            print("✅ Modelos descargados")
            print("✅ Analizador funcionando")
            print("✅ Listo para procesamiento")
            print("\n🚀 Ahora puedes ejecutar el procesamiento por lotes:")
            print("   python run_batch_processing.py \\")
            print("     --source /content/DanielaAS.jpg \\")
            print("     --videos /content/135.mp4 /content/136.mp4 /content/137.mp4 \\")
            print("     --output-dir /content/resultados \\")
            print("     --temp-frame-quality 100 \\")
            print("     --keep-fps")
            return True
        else:
            print("\n❌ Los modelos no funcionan correctamente")
            return False
    else:
        print("\n❌ Error en la descarga de modelos")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 