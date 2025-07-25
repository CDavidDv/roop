import cv2
import numpy
import onnxruntime
import threading
from typing import List, Any
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
            FACE_SWAPPER = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
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
    """Implementación simple del face swap"""
    try:
        # Obtener coordenadas de las caras
        source_bbox = source_face.bbox
        target_bbox = target_face.bbox
        
        # Extraer regiones de las caras
        source_x1, source_y1, source_x2, source_y2 = source_bbox
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        
        # Copiar la región de la cara fuente a la cara objetivo
        source_face_region = source_frame[source_y1:source_y2, source_x1:source_x2]
        target_face_region = target_frame[target_y1:target_y2, target_x1:target_x2]
        
        # Redimensionar para que coincidan
        if source_face_region.size > 0 and target_face_region.size > 0:
            resized_source = cv2.resize(source_face_region, (target_x2 - target_x1, target_y2 - target_y1))
            
            # Crear máscara para mezclar suavemente
            mask = numpy.ones((target_y2 - target_y1, target_x2 - target_x1), dtype=numpy.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Aplicar la cara fuente al frame objetivo
            target_frame[target_y1:target_y2, target_x1:target_x2] = (
                resized_source * mask[:, :, numpy.newaxis] +
                target_frame[target_y1:target_y2, target_x1:target_x2] * (1 - mask[:, :, numpy.newaxis])
            ).astype(numpy.uint8)
        
        return target_frame
        
    except Exception as e:
        print(f"Error en face swap: {e}")
        return target_frame

def process_frames(source_frames: List[Frame], target_frames: List[Frame], source_face: Face, target_face: Face) -> List[Frame]:
    """Procesa múltiples frames"""
    result_frames = []
    for target_frame in target_frames:
        swapped_frame = swap_face(source_face, target_face, source_frames[0], target_frame.copy())
        result_frames.append(swapped_frame)
    return result_frames

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Procesa una imagen"""
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

def process_frame(source_face: Face, target_frame: Frame) -> Frame:
    """Procesa un frame individual"""
    target_face = get_one_face(target_frame)
    
    if target_face:
        return swap_face(source_face, target_face, target_frame, target_frame.copy())
    else:
        return target_frame

NAME = 'ROOP.FACE_SWAPPER'
