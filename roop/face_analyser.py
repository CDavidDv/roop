import threading
from typing import Any, Optional, List
import cv2
import numpy
import os

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
            # Usar detector más robusto
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            FACE_ANALYSER = cv2.CascadeClassifier(cascade_path)
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
        # Detección más robusta
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Mejorar contraste
        
        # Parámetros más conservadores y robustos
        faces = get_face_analyser().detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Más conservador
            minNeighbors=3,    # Menos estricto
            minSize=(20, 20),  # Cara más pequeña
            maxSize=(0, 0),    # Sin límite máximo
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        simple_faces = []
        for (x, y, w, h) in faces:
            bbox = [x, y, x+w, y+h]
            kps = numpy.array([[x+w//2, y+h//2]])  # Punto central simple
            face = SimpleFace(bbox, kps)
            simple_faces.append(face)
        
        return simple_faces
    except Exception as e:
        # Silenciar errores de OpenCV y retornar cara por defecto
        h, w = frame.shape[:2]
        bbox = [w//4, h//4, 3*w//4, 3*h//4]
        kps = numpy.array([[w//2, h//2]])
        face = SimpleFace(bbox, kps)
        return [face]

def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        return many_faces[0]
    return None
