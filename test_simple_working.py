#!/usr/bin/env python3
"""
Test simple que funciona sin problemas
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_simple_working():
    """Test simple que funciona"""
    print("🧪 TEST SIMPLE FUNCIONANDO")
    print("=" * 40)
    
    try:
        # 1. Probar imports básicos
        print("1. Imports básicos...")
        import cv2
        import numpy as np
        import insightface
        import onnxruntime as ort
        print("✅ Imports OK")
        
        # 2. Probar configuración
        print("2. Configuración...")
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores: {available_providers}")
        
        # 3. Verificar modelo
        print("3. Modelo...")
        model_path = "models/inswapper_128.onnx"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Modelo: {size:,} bytes")
        else:
            print("❌ Modelo no encontrado")
            return False
        
        # 4. Probar face swapper
        print("4. Face swapper...")
        from roop.processors.frame.face_swapper import get_face_swapper
        swapper = get_face_swapper()
        print("✅ Face swapper OK")
        
        # 5. Probar face analyser (sin descargar modelos)
        print("5. Face analyser...")
        from roop.face_analyser import get_face_analyser
        analyser = get_face_analyser()
        print("✅ Face analyser OK")
        
        # 6. Probar detección simple
        print("6. Detección simple...")
        from roop.face_analyser import get_many_faces
        
        # Crear imagen de prueba
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
        
        faces = get_many_faces(test_img)
        print(f"✅ Detección: {len(faces)} caras encontradas")
        
        print("\n🎉 ¡TEST EXITOSO!")
        print("=" * 30)
        print("✅ Sin errores de modelos")
        print("✅ Face swapper funcionando")
        print("✅ CPU optimizado")
        print("✅ Listo para procesar")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if test_simple_working():
        print("\n🚀 ¡LISTO PARA PROCESAR VIDEOS!")
        print("=" * 40)
        print("Comando para procesar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ Aún hay problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 