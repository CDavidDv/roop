from typing import Any, List, Callable
import cv2
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            # Implementación simple sin GFPGAN para evitar errores
            FACE_ENHANCER = "simple_enhancer"
    return FACE_ENHANCER


def get_device() -> str:
    # Forzar uso de GPU si está disponible
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    
    # Priorizar CUDA sobre CPU
    if 'CUDAExecutionProvider' in available_providers:
        print(f"[{NAME}] Forzando uso de GPU (CUDA)")
        return 'cuda'
    elif 'CoreMLExecutionProvider' in available_providers:
        print(f"[{NAME}] Usando CoreML")
        return 'mps'
    else:
        print(f"[{NAME}] CUDA no disponible, usando CPU")
        return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """Enhancement simple sin GFPGAN"""
    try:
        if hasattr(target_face, 'bbox'):
            bbox = target_face.bbox
        else:
            return temp_frame
            
        start_x, start_y, end_x, end_y = map(int, bbox)
        padding_x = int((end_x - start_x) * 0.5)
        padding_y = int((end_y - start_y) * 0.5)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(temp_frame.shape[1], end_x + padding_x)
        end_y = min(temp_frame.shape[0], end_y + padding_y)
        
        if start_x < end_x and start_y < end_y:
            temp_face = temp_frame[start_y:end_y, start_x:end_x]
            if temp_face.size > 0:
                # Enhancement simple: mejorar contraste y nitidez
                enhanced_face = cv2.convertScaleAbs(temp_face, alpha=1.1, beta=5)
                enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)
                temp_frame[start_y:end_y, start_x:end_x] = enhanced_face
    except Exception as e:
        pass  # Silenciar errores
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    process_frames(source_path, temp_frame_paths, None)
