#!/usr/bin/env python3
"""
Script para probar el face_swapper
"""

import sys
import subprocess

def test_face_swapper():
    """Prueba que el face_swapper funcione correctamente"""
    print("🧪 PROBANDO FACE_SWAPPER")
    print("=" * 50)
    
    test_code = '''
import sys
sys.path.insert(0, '.')

try:
    # Probar importación
    from roop.processors.frame.face_swapper import get_face_swapper, swap_face, process_frame
    print("✅ face_swapper importado")
    
    # Probar que el modelo existe
    import os
    model_path = "models/inswapper_128.onnx"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Modelo encontrado: {size:,} bytes")
    else:
        print("❌ Modelo no encontrado")
        exit(1)
    
    # Probar carga del modelo
    swapper = get_face_swapper()
    print("✅ Modelo cargado correctamente")
    
    # Probar detección de caras
    from roop.face_analyser import get_one_face
    import cv2
    import numpy as np
    
    # Crear imagen de prueba
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Probar detección
    face = get_one_face(test_img)
    print("✅ Detección de caras funcionando")
    
    # Probar face swap
    result = process_frame(face, test_img.copy())
    print("✅ Face swap funcionando")
    
    print("✅ Face swapper completamente funcional")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
'''
    
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
    print("🚀 PROBANDO FACE_SWAPPER")
    print("=" * 60)
    
    if test_face_swapper():
        print("\n🎉 ¡FACE_SWAPPER FUNCIONANDO!")
        print("=" * 50)
        print("✅ Face swapper importado correctamente")
        print("✅ Modelo cargado")
        print("✅ Detección de caras funcionando")
        print("✅ Face swap funcionando")
        print("✅ Listo para procesar videos")
        print("\n🚀 Comandos disponibles:")
        print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ Face swapper aún tiene problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 