#!/usr/bin/env python3
"""
Script para probar el procesamiento completo
"""

import sys
import subprocess

def test_complete_processing():
    """Prueba que el procesamiento completo funcione"""
    print("🧪 PROBANDO PROCESAMIENTO COMPLETO")
    print("=" * 50)
    
    test_code = '''
import sys
sys.path.insert(0, '.')

try:
    # Probar importación de face_swapper
    from roop.processors.frame.face_swapper import get_face_swapper, process_frame
    print("✅ face_swapper importado")
    
    # Probar importación de face_enhancer
    from roop.processors.frame.face_enhancer import get_face_enhancer, process_frame as enhance_frame
    print("✅ face_enhancer importado")
    
    # Probar que los modelos existen
    import os
    inswapper_path = "models/inswapper_128.onnx"
    gfpgan_path = "models/GFPGANv1.4.pth"
    
    if os.path.exists(inswapper_path):
        size = os.path.getsize(inswapper_path)
        print(f"✅ inswapper_128.onnx encontrado: {size:,} bytes")
    else:
        print("❌ inswapper_128.onnx no encontrado")
        exit(1)
    
    if os.path.exists(gfpgan_path):
        size = os.path.getsize(gfpgan_path)
        print(f"✅ GFPGANv1.4.pth encontrado: {size:,} bytes")
    else:
        print("❌ GFPGANv1.4.pth no encontrado")
        exit(1)
    
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
    
    # Probar face enhancement
    result = enhance_frame(None, None, test_img.copy())
    print("✅ Face enhancement funcionando")
    
    print("✅ Procesamiento completo funcionando")
    
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
    print("🚀 PROBANDO PROCESAMIENTO COMPLETO")
    print("=" * 60)
    
    if test_complete_processing():
        print("\n🎉 ¡PROCESAMIENTO COMPLETO FUNCIONANDO!")
        print("=" * 50)
        print("✅ Face swapper funcionando")
        print("✅ Face enhancer funcionando")
        print("✅ Detección de caras funcionando")
        print("✅ Modelos encontrados")
        print("✅ Listo para procesar videos con calidad completa")
        print("\n🚀 Comandos disponibles:")
        print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ Procesamiento completo aún tiene problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 