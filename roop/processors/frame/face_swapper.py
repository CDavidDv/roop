import cv2
import numpy
import onnxruntime
import threading
from typing import List, Any, Callable
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
            
            # Configuración agresiva para GPU
            available_providers = ort.get_available_providers()
            print(f"[{NAME}] Proveedores disponibles: {available_providers}")
            
            # Configuración optimizada para GPU
            provider_options = []
            
            # Priorizar CUDA con configuración agresiva
            if 'CUDAExecutionProvider' in available_providers:
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                    'cudnn_conv_use_max_workspace': '1',
                    'do_copy_in_default_stream': '1',
                }
                provider_options = [
                    ('CUDAExecutionProvider', cuda_options),
                    ('CPUExecutionProvider', {
                        'intra_op_num_threads': 64,
                        'inter_op_num_threads': 64,
                    })
                ]
                print(f"[{NAME}] ✅ Configurando GPU agresivo con CUDA")
                print(f"[{NAME}] Opciones CUDA: {cuda_options}")
            else:
                # Fallback a CPU con optimizaciones
                provider_options = [
                    ('CPUExecutionProvider', {
                        'intra_op_num_threads': 64,
                        'inter_op_num_threads': 64,
                    })
                ]
                print(f"[{NAME}] ❌ CUDA no disponible, usando CPU optimizado")
            
            # Crear sesión ONNX optimizada
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.intra_op_num_threads = 64
            session_options.inter_op_num_threads = 64
            
            print(f"[{NAME}] Configurando sesión ONNX agresiva")
            print(f"[{NAME}] Hilos de ejecución: 64")
            
            # Cargar modelo con configuración agresiva
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
    # Optimización: convertir a float32 para mejor rendimiento GPU
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
