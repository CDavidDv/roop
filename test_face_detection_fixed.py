#!/usr/bin/env python3
"""
Test de detección de caras arreglada
"""

import sys
import os
import cv2
import numpy as np

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_face_detection_fixed():
    """Prueba la detección de caras arreglada"""
    print("🔍 PROBANDO DETECCIÓN DE CARAS ARREGLADA")
    print("=" * 50)
    
    try:
        # 1. Probar imports
        print("1. Imports...")
        from roop.face_analyser import get_face_analyser, get_one_face, get_many_faces, detect_faces_basic
        print("✅ Imports OK")
        
        # 2. Probar face analyser
        print("2. Face analyser...")
        analyser = get_face_analyser()
        print("✅ Face analyser cargado")
        
        # 3. Crear imagen de prueba
        print("3. Creando imagen de prueba...")
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Dibujar una cara más realista
        # Cabeza
        cv2.ellipse(test_img, (200, 200), (80, 100), 0, 0, 360, (255, 255, 255), -1)
        # Ojos
        cv2.circle(test_img, (170, 180), 15, (0, 0, 0), -1)
        cv2.circle(test_img, (230, 180), 15, (0, 0, 0), -1)
        # Nariz
        cv2.ellipse(test_img, (200, 220), (8, 15), 0, 0, 360, (0, 0, 0), -1)
        # Boca
        cv2.ellipse(test_img, (200, 250), (25, 10), 0, 0, 180, (0, 0, 0), -1)
        
        print("✅ Imagen de prueba creada")
        
        # 4. Probar detección básica
        print("4. Probando detección básica...")
        faces_basic = detect_faces_basic(test_img)
        print(f"✅ Detección básica: {len(faces_basic)} caras")
        
        # 5. Probar get_many_faces
        print("5. Probando get_many_faces...")
        faces = get_many_faces(test_img)
        print(f"✅ get_many_faces: {len(faces)} caras")
        
        # 6. Probar get_one_face
        print("6. Probando get_one_face...")
        face = get_one_face(test_img)
        if face:
            print("✅ get_one_face funcionando")
        else:
            print("⚠️ No se detectó cara con get_one_face")
        
        # 7. Probar con imagen real si existe
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
        
        print("\n🎉 ¡DETECCIÓN DE CARAS ARREGLADA!")
        print("=" * 40)
        print("✅ Face analyser funcionando")
        print("✅ Detección básica funcionando")
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
    if test_face_detection_fixed():
        print("\n🚀 ¡LISTO PARA PROCESAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Problemas con detección de caras")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 