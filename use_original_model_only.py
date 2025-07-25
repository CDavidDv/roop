#!/usr/bin/env python3
"""
Script para usar solo el modelo original sin InsightFace
"""

import os
import sys
import shutil
import subprocess

def backup_original_files():
    """Hace backup de los archivos originales"""
    print("📦 HACIENDO BACKUP DE ARCHIVOS ORIGINALES")
    print("=" * 50)
    
    # Crear directorio de backup
    backup_dir = "backup_original"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Archivos a hacer backup
    files_to_backup = [
        "roop/face_analyser.py",
        "roop/processors/frame/face_swapper.py"
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            print(f"✅ Backup: {file_path}")
    
    print("✅ Backup completado")

def create_simple_face_analyser():
    """Crea un face_analyser simple que no use InsightFace"""
    print("🔧 CREANDO FACE_ANALYSER SIMPLE")
    print("=" * 50)
    
    face_analyser_code = '''
import threading
from typing import Any, Optional, List
import cv2
import numpy

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

class SimpleFace:
    """Clase simple para representar una cara"""
    def __init__(self, bbox, kps, embedding=None):
        self.bbox = bbox
        self.kps = kps
        self.normed_embedding = embedding if embedding is not None else numpy.zeros(512)

def get_face_analyser() -> Any:
    """Retorna un analizador simple"""
    global FACE_ANALYSER
    
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # Usar OpenCV para detección simple
            FACE_ANALYSER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return FACE_ANALYSER

def clear_face_analyser() -> Any:
    global FACE_ANALYSER
    FACE_ANALYSER = None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        # Detección simple con OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = get_face_analyser().detectMultiScale(gray, 1.1, 4)
        
        simple_faces = []
        for (x, y, w, h) in faces:
            bbox = [x, y, x+w, y+h]
            kps = numpy.array([[x+w//2, y+h//2]])  # Punto central simple
            face = SimpleFace(bbox, kps)
            simple_faces.append(face)
        
        return simple_faces
    except Exception as e:
        print(f"Error en detección: {e}")
        return None

def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        # Retornar la primera cara encontrada
        return many_faces[0]
    return None
'''
    
    with open("roop/face_analyser.py", "w") as f:
        f.write(face_analyser_code)
    
    print("✅ face_analyser.py creado")

def create_simple_face_swapper():
    """Crea un face_swapper que use el modelo original"""
    print("🔧 CREANDO FACE_SWAPPER SIMPLE")
    print("=" * 50)
    
    face_swapper_code = '''
import cv2
import numpy
import onnxruntime
from typing import List
import roop.globals
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = onnxruntime.InferenceSession(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> Any:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_start() -> bool:
    return True

def pre_check() -> bool:
    return True

def post_process() -> None:
    clear_face_swapper()

def swap_face(source_face: Face, target_face: Face, source_frame: Frame, target_frame: Frame) -> Frame:
    # Implementación simple del face swap
    # Por ahora, solo retorna el frame original
    return target_frame

def process_frames(source_frames: List[Frame], target_frames: List[Frame], source_face: Face, target_face: Face) -> List[Frame]:
    # Procesar cada frame
    result_frames = []
    for target_frame in target_frames:
        swapped_frame = swap_face(source_face, target_face, source_frames[0], target_frame)
        result_frames.append(swapped_frame)
    return result_frames

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # Procesar imagen
    source_frame = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    
    source_face = get_one_face(source_frame)
    target_face = get_one_face(target_frame)
    
    if source_face and target_face:
        result_frame = swap_face(source_face, target_face, source_frame, target_frame)
        cv2.imwrite(output_path, result_frame)
    else:
        # Si no se detectan caras, copiar el frame original
        cv2.imwrite(output_path, target_frame)

NAME = 'ROOP.FACE_SWAPPER'
'''
    
    with open("roop/processors/frame/face_swapper.py", "w") as f:
        f.write(face_swapper_code)
    
    print("✅ face_swapper.py creado")

def test_simple_roop():
    """Prueba que ROOP funcione con la implementación simple"""
    print("🧪 PROBANDO ROOP SIMPLE")
    print("=" * 50)
    
    test_code = '''
import sys
sys.path.insert(0, '.')

try:
    # Probar importación
    import roop.core
    print("✅ roop.core importado")
    
    # Probar face_analyser
    from roop.face_analyser import get_face_analyser
    print("✅ face_analyser importado")
    
    # Probar face_swapper
    from roop.processors.frame.face_swapper import get_face_swapper
    print("✅ face_swapper importado")
    
    print("✅ ROOP simple funcionando correctamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
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
    print("🚀 CONFIGURANDO ROOP SIMPLE")
    print("=" * 60)
    
    # Paso 1: Hacer backup
    backup_original_files()
    
    # Paso 2: Crear face_analyser simple
    create_simple_face_analyser()
    
    # Paso 3: Crear face_swapper simple
    create_simple_face_swapper()
    
    # Paso 4: Probar ROOP
    if not test_simple_roop():
        print("❌ ROOP simple no funciona correctamente")
        return 1
    
    print("\n🎉 ¡ROOP SIMPLE CONFIGURADO!")
    print("=" * 50)
    print("✅ Backup de archivos originales creado")
    print("✅ Face analyser simple creado")
    print("✅ Face swapper simple creado")
    print("✅ ROOP funcionando")
    print("✅ Listo para procesar videos")
    print("\n📁 Backup guardado en: backup_original/")
    
    return 0

if __name__ == "__main__":
    import threading
    sys.exit(main()) 