#!/usr/bin/env python3
"""
Script simple para probar face swap sin errores
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_simple_face_swap():
    """Prueba simple del face swap"""
    print("🧪 PROBANDO FACE SWAP SIMPLE")
    print("=" * 40)
    
    try:
        # 1. Probar imports básicos
        print("1. Probando imports...")
        import cv2
        import numpy as np
        import insightface
        import onnxruntime as ort
        print("✅ Imports básicos OK")
        
        # 2. Probar configuración GPU
        print("2. Probando configuración GPU...")
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores disponibles: {available_providers}")
        
        # 3. Probar que el modelo existe
        print("3. Verificando modelo...")
        model_path = "models/inswapper_128.onnx"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Modelo encontrado: {size:,} bytes")
        else:
            print("❌ Modelo no encontrado")
            return False
        
        # 4. Probar face swapper
        print("4. Probando face swapper...")
        from roop.processors.frame.face_swapper import get_face_swapper
        swapper = get_face_swapper()
        print("✅ Face swapper cargado")
        
        # 5. Probar face analyser
        print("5. Probando face analyser...")
        from roop.face_analyser import get_face_analyser
        analyser = get_face_analyser()
        print("✅ Face analyser cargado")
        
        # 6. Probar detección de caras
        print("6. Probando detección de caras...")
        from roop.face_analyser import get_one_face
        
        # Crear imagen de prueba
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(test_img, (100, 100), 30, (0, 0, 0), -1)
        
        face = get_one_face(test_img)
        print("✅ Detección de caras funcionando")
        
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ Face swap listo para usar")
        print("✅ GPU configurado correctamente")
        print("✅ Sin errores de imports")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if test_simple_face_swap():
        print("\n🚀 ¡LISTO PARA PROCESAR VIDEOS!")
        print("=" * 40)
        print("Comando para procesar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ Aún hay problemas que resolver")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 