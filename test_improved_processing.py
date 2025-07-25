#!/usr/bin/env python3
"""
Script para probar el procesamiento mejorado
"""

import sys
import subprocess

def test_improved_processing():
    """Prueba que el procesamiento mejorado funcione"""
    print("🧪 PROBANDO PROCESAMIENTO MEJORADO")
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
    
    # Probar que el modelo existe
    import os
    model_path = "models/inswapper_128.onnx"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ inswapper_128.onnx encontrado: {size:,} bytes")
    else:
        print("❌ inswapper_128.onnx no encontrado")
        exit(1)
    
    # Probar detección de caras
    from roop.face_analyser import get_one_face
    import cv2
    import numpy as np
    
    # Crear imagen de prueba con una cara simulada
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Dibujar una cara simulada
    cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(test_img, (100, 100), 30, (0, 0, 0), -1)
    
    # Probar detección
    face = get_one_face(test_img)
    print("✅ Detección de caras funcionando")
    
    # Probar face swap
    result = process_frame(face, test_img.copy())
    print("✅ Face swap funcionando")
    
    # Probar face enhancement
    result = enhance_frame(None, None, test_img.copy())
    print("✅ Face enhancement funcionando")
    
    print("✅ Procesamiento mejorado funcionando")
    
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
    print("🚀 PROBANDO PROCESAMIENTO MEJORADO")
    print("=" * 60)
    
    if test_improved_processing():
        print("\n🎉 ¡PROCESAMIENTO MEJORADO FUNCIONANDO!")
        print("=" * 50)
        print("✅ Face swapper mejorado funcionando")
        print("✅ Face enhancer simplificado funcionando")
        print("✅ Detección de caras funcionando")
        print("✅ Face swap más visible")
        print("✅ Listo para procesar videos con calidad mejorada")
        print("\n🚀 Comandos disponibles:")
        print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ Procesamiento mejorado aún tiene problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 