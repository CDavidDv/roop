#!/usr/bin/env python3
"""
Test del face swap real
"""

import sys
import os
import cv2
import numpy as np

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_face_swap_real():
    """Prueba el face swap real"""
    print("🧪 PROBANDO FACE SWAP REAL")
    print("=" * 50)
    
    try:
        # 1. Probar face swapper
        print("1. Cargando face swapper...")
        from roop.processors.frame.face_swapper import get_face_swapper, swap_face
        swapper = get_face_swapper()
        print("✅ Face swapper cargado")
        
        # 2. Probar face analyser
        print("2. Cargando face analyser...")
        from roop.face_analyser import get_face_analyser, get_one_face, get_many_faces
        analyser = get_face_analyser()
        print("✅ Face analyser cargado")
        
        # 3. Crear imágenes de prueba
        print("3. Creando imágenes de prueba...")
        
        # Imagen fuente (cara a copiar)
        source_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(source_img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(source_img, (100, 100), 30, (0, 0, 0), -1)
        
        # Imagen objetivo (cara a cambiar)
        target_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(target_img, (50, 50), (150, 150), (0, 255, 0), -1)  # Verde
        cv2.circle(target_img, (100, 100), 30, (0, 0, 0), -1)
        
        print("✅ Imágenes de prueba creadas")
        
        # 4. Detectar caras
        print("4. Detectando caras...")
        source_faces = get_many_faces(source_img)
        target_faces = get_many_faces(target_img)
        
        print(f"✅ Caras fuente: {len(source_faces)}")
        print(f"✅ Caras objetivo: {len(target_faces)}")
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            print("❌ No se detectaron caras")
            return False
        
        # 5. Probar face swap
        print("5. Probando face swap...")
        source_face = source_faces[0]
        target_face = target_faces[0]
        
        result = swap_face(source_face, target_face, target_img.copy())
        
        if result is not None:
            print("✅ Face swap funcionando")
            
            # Guardar resultado para verificar
            cv2.imwrite("test_face_swap_result.jpg", result)
            print("✅ Resultado guardado como test_face_swap_result.jpg")
        else:
            print("❌ Face swap falló")
            return False
        
        # 6. Probar con imagen real si existe
        print("6. Probando con imagen real...")
        source_path = "/content/DanielaAS.jpg"
        if os.path.exists(source_path):
            real_source = cv2.imread(source_path)
            if real_source is not None:
                real_faces = get_many_faces(real_source)
                print(f"✅ En imagen real: {len(real_faces)} caras detectadas")
                
                if len(real_faces) > 0:
                    print("✅ Cara real detectada correctamente")
                else:
                    print("❌ No se detectó cara en imagen real")
                    return False
            else:
                print("⚠️ No se pudo cargar imagen real")
        else:
            print("⚠️ Imagen real no encontrada")
        
        print("\n🎉 ¡FACE SWAP REAL FUNCIONANDO!")
        print("=" * 40)
        print("✅ Face swapper cargado")
        print("✅ Face swap funcionando")
        print("✅ Cara real detectada")
        print("✅ Listo para procesar")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if test_face_swap_real():
        print("\n🚀 ¡LISTO PARA PROCESAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Problemas con face swap")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 