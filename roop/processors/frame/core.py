import os
import sys
import importlib
import psutil
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import roop

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_frames',
    'process_image',
    'process_video',
    'post_process'
]


def clear_gpu_memory():
    """Liberar memoria GPU y limpiar cachés"""
    try:
        # Limpiar caché de PyTorch
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[MEMORY] PyTorch GPU cache liberado")
    except ImportError:
        pass
    
    try:
        # Limpiar caché de TensorFlow
        import tensorflow as tf
        tf.keras.backend.clear_session()
        print("[MEMORY] TensorFlow session liberada")
    except ImportError:
        pass
    
    # Forzar garbage collection
    gc.collect()
    print("[MEMORY] Garbage collection ejecutado")


def wait_for_gpu_memory_release(wait_time: int = 30):
    """Esperar y monitorear liberación de memoria GPU"""
    print(f"[MEMORY] Esperando {wait_time} segundos para liberar memoria GPU...")
    
    for i in range(wait_time, 0, -1):
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[MEMORY] {i}s restantes - VRAM: {memory_allocated:.2f}GB usado, {memory_reserved:.2f}GB reservado", end='\r')
        except:
            print(f"[MEMORY] {i}s restantes...", end='\r')
        time.sleep(1)
    
    print("\n[MEMORY] Pausa completada")
    clear_gpu_memory()


def process_video_with_memory_management(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None], processor_name: str = "unknown") -> None:
    """Procesar video con gestión de memoria entre procesadores"""
    print(f"[{processor_name.upper()}] Iniciando procesamiento...")
    
    # Formato más compacto para la barra de progreso
    progress_bar_format = '{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    total = len(frame_paths)
    
    with tqdm(total=total, desc=f'Processing {processor_name}', unit='frame', 
              dynamic_ncols=True, bar_format=progress_bar_format, 
              leave=False, position=0) as progress:
        # Crear una función de actualización que pase el número de frame
        def update_with_frame(frame_index):
            update_progress(progress, processor_name, frame_index + 1, total)
        
        # Procesar frames con información detallada
        for i, frame_path in enumerate(frame_paths):
            # Procesar un solo frame para mostrar progreso detallado
            process_frames(source_path, [frame_path], lambda: update_with_frame(i))
    
    print(f"[{processor_name.upper()}] Procesamiento completado")
    
    # Liberar memoria después de cada procesador
    clear_gpu_memory()
    
    # Pausa entre procesadores para liberar memoria
    if processor_name.lower() != "face_enhancer":  # No pausar después del último procesador
        wait_for_gpu_memory_release(roop.globals.gpu_memory_wait_time)  # Pausa configurable


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'roop.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError as e:
        print(f'Frame processor {frame_processor} not found.')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except NotImplementedError:
        print(f'Frame processor {frame_processor} not implemented correctly.')
        sys.exit(1)
    except Exception as e:
        print(f'Error importando frame processor {frame_processor}: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None]) -> None:
    with ThreadPoolExecutor(max_workers=roop.globals.execution_threads) as executor:
        futures = []
        queue = create_queue(temp_frame_paths)
        queue_per_future = max(len(temp_frame_paths) // roop.globals.execution_threads, 1)
        while not queue.empty():
            future = executor.submit(process_frames, source_path, pick_queue(queue, queue_per_future), update)
            futures.append(future)
        for future in as_completed(futures):
            future.result()


def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    # Formato más compacto para la barra de progreso
    progress_bar_format = '{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', 
              dynamic_ncols=True, bar_format=progress_bar_format,
              leave=False, position=0) as progress:
        multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress, "unknown", 0, total))


def update_progress(progress: Any = None, processor_name: str = "unknown", frame_number: int = 0, total_frames: int = 0) -> None:
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    
    # Información de GPU si está disponible
    gpu_info = ""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_info = f"GPU:{gpu_memory:.1f}GB"
    except:
        pass
    
    # Calcular porcentaje
    if total_frames > 0:
        percentage = (frame_number / total_frames) * 100
        progress_text = f"{frame_number}/{total_frames} ({percentage:.1f}%)"
    else:
        progress_text = f"{frame_number}"
    
    # Crear postfix más compacto
    postfix_parts = []
    postfix_parts.append(f"RAM:{memory_usage:.1f}GB")
    if gpu_info:
        postfix_parts.append(gpu_info)
    postfix_parts.append(f"Threads:{roop.globals.execution_threads}")
    
    progress.set_postfix({
        'processor': processor_name,
        'frame': progress_text,
        'info': ' | '.join(postfix_parts)
    })
    progress.refresh()
    progress.update(1)
