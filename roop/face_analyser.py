import threading
from typing import Any, Optional, List
import insightface
import numpy
import os

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # Configuración que funcione
            try:
                print("🔄 Inicializando face analyser...")
                # Usar configuración que funcione
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
                FACE_ANALYSER.prepare(ctx_id=0)
                print("✅ Face analyser cargado correctamente")
            except Exception as e:
                print(f"⚠️ Error cargando face analyser: {e}")
                try:
                    # Segundo intento sin nombre específico
                    FACE_ANALYSER = insightface.app.FaceAnalysis()
                    FACE_ANALYSER.prepare(ctx_id=0)
                    print("✅ Face analyser cargado con configuración simple")
                except Exception as e2:
                    print(f"❌ Error crítico en face analyser: {e2}")
                    print("⚠️ Usando fallback para face analyser")
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
            # Fallback simple si no hay analyser
            print("⚠️ No hay face analyser disponible")
            return []
        
        faces = analyser.get(frame)
        print(f"🔍 Detectadas {len(faces)} caras en el frame")
        return faces
    except Exception as e:
        print(f"⚠️ Error en detección de caras: {e}")
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
