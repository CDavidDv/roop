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
