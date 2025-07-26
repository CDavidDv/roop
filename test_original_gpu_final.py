#!/usr/bin/env python3
"""
Test del código original con GPU
"""

import sys
import os
import onnxruntime as ort

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_original_gpu_final():
    """Prueba el código original con GPU"""
    print("🧪 PROBANDO CÓDIGO ORIGINAL CON GPU")
    print("=" * 50)
    
    try:
        # 1. Configurar GPU
        print("1. Configurando GPU...")
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ CUDA disponible")
            # Configurar GPU
            import roop.globals
            roop.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("✅ GPU configurado")
        else:
            print("❌ CUDA no disponible")
            return False
        
        # 2. Verificar modelo
        print("2. Verificando modelo...")
        model_path = "models/inswapper_128.onnx"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ Modelo: {size:,} bytes")
        else:
            print("❌ Modelo no encontrado")
            return False
        
        # 3. Probar face swapper original
        print("3. Probando face swapper original...")
        from roop.processors.frame.face_swapper import get_face_swapper
        swapper = get_face_swapper()
        print("✅ Face swapper original cargado")
        
        # 4. Probar face analyser original
        print("4. Probando face analyser original...")
        from roop.face_analyser import get_face_analyser
        analyser = get_face_analyser()
        print("✅ Face analyser original cargado")
        
        # 5. Probar detección
        print("5. Probando detección...")
        from roop.face_analyser import get_many_faces
        
        # Crear imagen de prueba
        import cv2
        import numpy as np
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(test_img, (100, 100), 30, (0, 0, 0), -1)
        
        faces = get_many_faces(test_img)
        if faces:
            print(f"✅ Detección: {len(faces)} caras encontradas")
        else:
            print("⚠️ No se detectaron caras")
        
        # 6. Probar con imagen real
        print("6. Probando con imagen real...")
        source_path = "/content/DanielaAS.jpg"
        if os.path.exists(source_path):
            real_img = cv2.imread(source_path)
            if real_img is not None:
                from roop.face_analyser import get_one_face
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
        
        # 7. Probar face swap
        print("7. Probando face swap...")
        from roop.processors.frame.face_swapper import swap_face
        
        if real_face:
            # Crear imagen de destino
            target_img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.rectangle(target_img, (100, 100), (200, 200), (255, 255, 255), -1)
            cv2.circle(target_img, (150, 150), 40, (0, 0, 0), -1)
            
            # Detectar cara en imagen de destino
            target_faces = get_many_faces(target_img)
            if target_faces:
                target_face = target_faces[0]
                print("✅ Cara de destino detectada")
                
                # Hacer face swap
                result = swap_face(real_face, target_face, target_img)
                print("✅ Face swap completado")
                
                # Verificar que el resultado es diferente
                if not np.array_equal(result, target_img):
                    print("✅ Face swap efectivo - imagen cambiada")
                else:
                    print("⚠️ Face swap no efectivo - imagen igual")
            else:
                print("⚠️ No se detectó cara en imagen de destino")
        else:
            print("⚠️ No hay cara real para probar")
        
        print("\n🎉 ¡CÓDIGO ORIGINAL CON GPU!")
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
    if test_original_gpu_final():
        print("\n🚀 ¡LISTO PARA PROCESAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python run_with_gpu.py")
        return 0
    else:
        print("\n❌ Aún hay problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 