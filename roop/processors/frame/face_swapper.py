from typing import Any, List, Callable
import cv2
import insightface
import threading
import onnxruntime as ort

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
            
            # Forzar uso de GPU si está disponible
            available_providers = ort.get_available_providers()
            
            print(f"[{NAME}] Proveedores disponibles: {available_providers}")
            
            # Estrategia 1: Intentar con configuración de sesión ONNX
            if 'CUDAExecutionProvider' in available_providers:
                print(f"[{NAME}] ✅ CUDA disponible, intentando forzar GPU...")
                
                # Configurar opciones de CUDA
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_use_max_workspace': '1',
                    'do_copy_in_default_stream': '1',
                }
                
                # Intentar diferentes configuraciones de proveedores
                provider_configs = [
                    # Configuración 1: Solo CUDA
                    (['CUDAExecutionProvider'], {'CUDAExecutionProvider': cuda_options}),
                    # Configuración 2: CUDA + CPU como fallback
                    (['CUDAExecutionProvider', 'CPUExecutionProvider'], {'CUDAExecutionProvider': cuda_options}),
                    # Configuración 3: TensorRT + CUDA
                    (['TensorrtExecutionProvider', 'CUDAExecutionProvider'], {'CUDAExecutionProvider': cuda_options}),
                ]
                
                for providers, provider_options in provider_configs:
                    try:
                        print(f"[{NAME}] Intentando con proveedores: {providers}")
                        
                        # Crear sesión ONNX con configuración específica
                        session_options = ort.SessionOptions()
                        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                        
                        FACE_SWAPPER = insightface.model_zoo.get_model(
                            model_path, 
                            providers=providers,
                            provider_options=provider_options,
                            session_options=session_options
                        )
                        
                        # Verificar si se aplicó CUDA
                        if hasattr(FACE_SWAPPER, 'providers'):
                            print(f"[{NAME}] ✅ Modelo cargado con proveedores: {FACE_SWAPPER.providers}")
                            if 'CUDAExecutionProvider' in FACE_SWAPPER.providers:
                                print(f"[{NAME}] 🎉 GPU forzado exitosamente!")
                                break
                        else:
                            print(f"[{NAME}] Modelo cargado (no se puede verificar proveedores)")
                            break
                            
                    except Exception as e:
                        print(f"[{NAME}] ❌ Error con configuración {providers}: {e}")
                        continue
                
                # Si todas las configuraciones fallaron, usar configuración por defecto
                if FACE_SWAPPER is None:
                    print(f"[{NAME}] ⚠️ Fallback a configuración por defecto")
                    FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
                    
            else:
                # Fallback a CPU si CUDA no está disponible
                providers = roop.globals.execution_providers
                print(f"[{NAME}] ❌ CUDA no disponible, usando: {providers}")
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
                
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
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


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
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
