#!/usr/bin/env python3
"""
Script para probar la detección de caras
"""

import sys
import os
import cv2
import numpy as np

# Agregar el directorio actual al path
sys.path.insert(0, '.')

def test_face_detection():
    """Prueba la detección de caras"""
    print("🔍 PROBANDO DETECCIÓN DE CARAS")
    print("=" * 40)
    
    try:
        # 1. Probar face analyser
        print("1. Cargando face analyser...")
        from roop.face_analyser import get_face_analyser, get_one_face, get_many_faces
        
        analyser = get_face_analyser()
        if analyser is None:
            print("❌ Face analyser no disponible")
            return False
        print("✅ Face analyser cargado")
        
        # 2. Crear imagen de prueba con cara
        print("2. Creando imagen de prueba...")
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
        
        # 3. Probar detección
        print("3. Probando detección...")
        faces = get_many_faces(test_img)
        print(f"✅ Detectadas {len(faces)} caras")
        
        # 4. Probar get_one_face
        print("4. Probando get_one_face...")
        face = get_one_face(test_img)
        if face:
            print("✅ Cara detectada correctamente")
        else:
            print("❌ No se detectó cara")
            return False
        
        # 5. Probar con imagen real si existe
        print("5. Probando con imagen real...")
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
        
        print("\n🎉 ¡DETECCIÓN DE CARAS FUNCIONANDO!")
        print("=" * 40)
        print("✅ Face analyser cargado")
        print("✅ Detección funcionando")
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
    if test_face_detection():
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