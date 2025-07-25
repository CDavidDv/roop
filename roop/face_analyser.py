import threading
from typing import Any, Optional, List
import insightface
import numpy
import cv2
import os

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # Verificar si existe el modelo original
            model_path = "models/inswapper_128.onnx"
            if os.path.exists(model_path):
                print("[FACE_ANALYSER] Usando modelo original inswapper_128.onnx")
                # Usar InsightFace con modelo personalizado
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                FACE_ANALYSER.prepare(ctx_id=0)
            else:
                # Fallback a InsightFace estándar
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                
                # Priorizar CUDA sobre CPU
                if 'CUDAExecutionProvider' in available_providers:
                    # Usar solo CUDA para forzar GPU
                    providers = ['CUDAExecutionProvider']
                    print("[FACE_ANALYSER] Forzando uso de GPU (CUDA)")
                else:
                    # Fallback a CPU si CUDA no está disponible
                    providers = roop.globals.execution_providers
                    print(f"[FACE_ANALYSER] CUDA no disponible, usando: {providers}")
                
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
                FACE_ANALYSER.prepare(ctx_id=0)
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
        return get_face_analyser().get(frame)
    except ValueError:
        return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
