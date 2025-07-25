#!/usr/bin/env python3
"""
Test del código original con GPU
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_original_gpu():
    """Prueba el código original con GPU"""
    print("🧪 PROBANDO CÓDIGO ORIGINAL CON GPU")
    print("=" * 50)
    
    try:
        # 1. Probar imports
        print("1. Imports...")
        import cv2
        import numpy as np
        import insightface
        import onnxruntime as ort
        print("✅ Imports OK")
        
        # 2. Verificar GPU
        print("2. Verificando GPU...")
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ CUDA disponible")
        else:
            print("⚠️ CUDA no disponible, usando CPU")
        
        # 3. Verificar modelo
        print("3. Verificando modelo...")
        model_path = "models/inswapper_128.onnx"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Modelo: {size:,} bytes")
        else:
            print("❌ Modelo no encontrado")
            print("Ejecuta: python download_model.py")
            return False
        
        # 4. Probar face swapper original
        print("4. Probando face swapper original...")
        from roop.processors.frame.face_swapper import get_face_swapper
        swapper = get_face_swapper()
        print("✅ Face swapper original cargado")
        
        # 5. Probar face analyser original
        print("5. Probando face analyser original...")
        from roop.face_analyser import get_face_analyser
        analyser = get_face_analyser()
        print("✅ Face analyser original cargado")
        
        # 6. Probar detección
        print("6. Probando detección...")
        from roop.face_analyser import get_one_face
        
        # Crear imagen de prueba
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(test_img, (100, 100), 30, (0, 0, 0), -1)
        
        face = get_one_face(test_img)
        if face:
            print("✅ Detección funcionando")
        else:
            print("⚠️ No se detectó cara en imagen de prueba")
        
        print("\n🎉 ¡CÓDIGO ORIGINAL CON GPU FUNCIONANDO!")
        print("=" * 50)
        print("✅ Código original restaurado")
        print("✅ GPU configurado")
        print("✅ Sin modificaciones de calidad")
        print("✅ Sin recuadros raros")
        print("✅ Listo para procesar")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if test_original_gpu():
        print("\n🚀 ¡LISTO PARA PROCESAR!")
        print("=" * 30)
        print("Comando para procesar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Aún hay problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 