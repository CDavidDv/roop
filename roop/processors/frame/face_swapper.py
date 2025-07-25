import cv2
import numpy
import onnxruntime as ort
import threading
from typing import List, Any, Callable
import roop.globals
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, conditional_download, is_image, is_video
import insightface

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER
    
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            
            # Configuración simplificada para evitar errores de CUDA
            print(f"[{NAME}] Configurando face swapper...")
            
            # Usar solo CPU por ahora para evitar errores de CUDA
            provider_options = [
                ('CPUExecutionProvider', {
                    'intra_op_num_threads': 64,
                    'inter_op_num_threads': 64,
                })
            ]
            
            print(f"[{NAME}] ✅ Usando CPU optimizado (64 hilos)")
            
            # Crear sesión ONNX optimizada
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.intra_op_num_threads = 64
            session_options.inter_op_num_threads = 64
            
            print(f"[{NAME}] Configurando sesión ONNX agresiva")
            print(f"[{NAME}] Hilos de ejecución: 64")
            
            # Cargar modelo con configuración simplificada
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path, 
                    providers=provider_options,
                    session_options=session_options
                )
                print(f"[{NAME}] ✅ Modelo cargado correctamente")
            except Exception as e:
                print(f"[{NAME}] ⚠️ Error cargando modelo: {e}")
                # Fallback más simple
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
                print(f"[{NAME}] ✅ Modelo cargado con configuración simple")
                
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # Optimización: convertir a float32 para mejor rendimiento
    if temp_frame.dtype != numpy.float32:
        temp_frame = temp_frame.astype(numpy.float32)
    
    result = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    
    # Asegurar que el resultado sea uint8 para OpenCV
    if result.dtype != numpy.uint8:
        result = numpy.clip(result, 0, 255).astype(numpy.uint8)
    
    return result


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
