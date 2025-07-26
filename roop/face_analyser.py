import threading
from typing import Any, Optional, List
import insightface
import numpy
import cv2

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            try:
                # Usar configuración simple sin descargar modelos adicionales
                FACE_ANALYSER = insightface.app.FaceAnalysis()
                FACE_ANALYSER.prepare(ctx_id=0)
                print("✅ Face analyser cargado con configuración simple")
            except Exception as e:
                print(f"⚠️ Error cargando face analyser: {e}")
                # Fallback - crear un mock
                FACE_ANALYSER = None
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER
    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces and len(many_faces) > 0:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        analyser = get_face_analyser()
        if analyser is None:
            # Fallback simple si no hay analyser - usar detección básica
            return detect_faces_basic(frame)
        return analyser.get(frame)
    except Exception as e:
        print(f"⚠️ Error en detección de caras: {e}")
        return detect_faces_basic(frame)


def detect_faces_basic(frame: Frame) -> List[Face]:
    """Detección básica de caras usando OpenCV"""
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Cargar clasificador de caras de OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detectar caras
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convertir a formato de Face
        face_objects = []
        for (x, y, w, h) in faces:
            # Crear objeto Face simple
            face_obj = type('Face', (), {
                'bbox': [x, y, x + w, y + h],
                'kps': None,
                'det_score': 0.9,
                'normed_embedding': numpy.zeros(512)  # Embedding dummy
            })()
            face_objects.append(face_obj)
        
        print(f"🔍 Detectadas {len(face_objects)} caras con OpenCV")
        return face_objects
        
    except Exception as e:
        print(f"⚠️ Error en detección básica: {e}")
        return []


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
