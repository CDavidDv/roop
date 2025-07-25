from typing import Any, List, Callable
import cv2
import insightface
import threading
import onnxruntime as ort
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../inswapper_128.onnx')
            
            # Configuración optimizada para GPU
            available_providers = ort.get_available_providers()
            print(f"[{NAME}] Proveedores disponibles: {available_providers}")
            
            # Configuración optimizada para GPU
            provider_options = []
            
            # Priorizar CUDA con configuración optimizada
            if 'CUDAExecutionProvider' in available_providers:
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_use_max_workspace': '1',
                    'do_copy_in_default_stream': '1',
                }
                provider_options = [
                    ('CUDAExecutionProvider', cuda_options),
                    ('CPUExecutionProvider', {})
                ]
                print(f"[{NAME}] ✅ Configurando GPU optimizado con CUDA")
                print(f"[{NAME}] Opciones CUDA: {cuda_options}")
            else:
                # Fallback a CPU con optimizaciones
                provider_options = [
                    ('CPUExecutionProvider', {
                        'intra_op_num_threads': roop.globals.execution_threads,
                        'inter_op_num_threads': roop.globals.execution_threads,
                    })
                ]
                print(f"[{NAME}] ❌ CUDA no disponible, usando CPU optimizado")
            
            # Crear sesión ONNX optimizada
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.intra_op_num_threads = roop.globals.execution_threads
            session_options.inter_op_num_threads = roop.globals.execution_threads
            
            print(f"[{NAME}] Configurando sesión ONNX optimizada")
            print(f"[{NAME}] Hilos de ejecución: {roop.globals.execution_threads}")
            
            # Cargar modelo con configuración optimizada
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, 
                providers=provider_options,
                session_options=session_options
            )
            
            # Verificar configuración aplicada
            if hasattr(FACE_SWAPPER, 'providers'):
                print(f"[{NAME}] Modelo cargado con proveedores: {FACE_SWAPPER.providers}")
            else:
                print(f"[{NAME}] Modelo cargado (no se puede verificar proveedores)")
                
    return FACE_SWAPPER


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../content/roop')
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
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
    global FACE_SWAPPER

    FACE_SWAPPER = None


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # Optimización: convertir a float32 para mejor rendimiento GPU
    if temp_frame.dtype != np.float32:
        temp_frame = temp_frame.astype(np.float32)
    
    result = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    
    # Asegurar que el resultado sea uint8 para OpenCV
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        if many_faces := get_many_faces(temp_frame):
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    elif target_face := get_one_face(temp_frame):
        temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    process_frames(source_path, temp_frame_paths, roop.processors.frame.core.update_progress)
