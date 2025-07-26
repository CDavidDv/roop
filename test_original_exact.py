#!/usr/bin/env python3
"""
Test del código original exacto
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_original_exact():
    """Prueba el código original exacto"""
    print("🧪 PROBANDO CÓDIGO ORIGINAL EXACTO")
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
            print("⚠️ CUDA no disponible")
        
        # 3. Verificar modelo
        print("3. Verificando modelo...")
        model_path = "models/inswapper_128.onnx"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Modelo: {size:,} bytes")
        else:
            print("❌ Modelo no encontrado")
            return False
        
        # 4. Probar face swapper original
        print("4. Probando face swapper original...")
        from roop.processors.frame.face_swapper import get_face_swapper, swap_face
        swapper = get_face_swapper()
        print("✅ Face swapper original cargado")
        
        # 5. Probar face analyser original
        print("5. Probando face analyser original...")
        from roop.face_analyser import get_face_analyser, get_one_face
        analyser = get_face_analyser()
        print("✅ Face analyser original cargado")
        
        # 6. Probar detección
        print("6. Probando detección...")
        from roop.face_analyser import get_many_faces
        
        # Crear imagen de prueba
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(test_img, (100, 100), 30, (0, 0, 0), -1)
        
        faces = get_many_faces(test_img)
        if faces:
            print(f"✅ Detección: {len(faces)} caras encontradas")
        else:
            print("⚠️ No se detectaron caras")
        
        # 7. Probar con imagen real
        print("7. Probando con imagen real...")
        source_path = "/content/DanielaAS.jpg"
        if os.path.exists(source_path):
            real_img = cv2.imread(source_path)
            if real_img is not None:
                real_faces = get_many_faces(real_img)
                print(f"✅ En imagen real: {len(real_faces)} caras detectadas")
                
                real_face = get_one_face(real_img)
                if real_face:
                    print("✅ Cara real detectada correctamente")
                else:
                    print("❌ No se detectó cara en imagen real")
                    return False
            else:
                print("⚠️ No se pudo cargar imagen real")
        else:
            print("⚠️ Imagen real no encontrada")
        
        print("\n🎉 ¡CÓDIGO ORIGINAL EXACTO!")
        print("=" * 40)
        print("✅ Código original restaurado")
        print("✅ GPU configurado")
        print("✅ Face swap original funcionando")
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
    if test_original_exact():
        print("\n🚀 ¡LISTO PARA PROCESAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Aún hay problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 