import os
import sys
import importlib
import psutil
import time
import gc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import roop

# Suprimir warnings
warnings.filterwarnings('ignore')

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


def get_optimal_thread_count():
    """Obtener número óptimo de hilos según el sistema"""
    cpu_count = psutil.cpu_count(logical=True)
    
    # Verificar si hay GPU disponible
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    if gpu_available:
        # Con GPU, usar menos hilos para evitar saturación
        return min(16, cpu_count // 2)
    else:
        # Sin GPU, usar más hilos para CPU
        return min(31, cpu_count - 1)


def process_video_with_memory_management(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None], processor_name: str = "unknown") -> None:
    """Procesar video con gestión de memoria entre procesadores"""
    print(f"[{processor_name.upper()}] Iniciando procesamiento optimizado...")
    
    # Formato más compacto para la barra de progreso
    progress_bar_format = '{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    total = len(frame_paths)
    
    # Obtener número óptimo de hilos
    optimal_threads = get_optimal_thread_count()
    execution_threads = roop.globals.execution_threads if roop.globals.execution_threads is not None else optimal_threads
    
    print(f"[{processor_name.upper()}] Usando {execution_threads} hilos para procesamiento paralelo")
    
    with tqdm(total=total, desc=f'Processing {processor_name}', unit='frame', 
              dynamic_ncols=True, bar_format=progress_bar_format, 
              leave=False, position=0) as progress:
        
        # Usar procesamiento paralelo optimizado
        with ThreadPoolExecutor(max_workers=execution_threads) as executor:
            futures = []
            queue = create_queue(frame_paths)
            queue_per_future = max(len(frame_paths) // execution_threads, 1)
            
            # Crear función de actualización optimizada
            def update_with_frame(frame_index):
                update_progress(progress, processor_name, frame_index + 1, total)
            
            # Procesar frames en paralelo con gestión de memoria
            while not queue.empty():
                batch_frames = pick_queue(queue, queue_per_future)
                future = executor.submit(process_frames, source_path, batch_frames, lambda: update_with_frame(len(futures)))
                futures.append(future)
            
            # Esperar a que terminen todos los futures
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[{processor_name.upper()}] Error en procesamiento: {e}")
            
            # Liberar memoria después del procesamiento
            if roop.globals.gpu_memory_wait_time > 0:
                wait_for_gpu_memory_release(roop.globals.gpu_memory_wait_time)


def load_frame_processor_module(frame_processor: str) -> Any:
    frame_processor_module = importlib.import_module('roop.processors.frame.' + frame_processor)
    met_requirements = all(hasattr(frame_processor_module, frame_processor_interface) for frame_processor_interface in FRAME_PROCESSORS_INTERFACE)
    if not met_requirements:
        raise NotImplementedError
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    frame_processors_modules = []
    for frame_processor in frame_processors:
        frame_processor_module = load_frame_processor_module(frame_processor)
        frame_processors_modules.append(frame_processor_module)
    return frame_processors_modules


def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None]) -> None:
    # Asegurar que execution_threads tenga un valor válido
    if roop.globals.execution_threads is None:
        roop.globals.execution_threads = get_optimal_thread_count()
    
    # Usar procesamiento optimizado con gestión de memoria
    process_video_with_memory_management(source_path, temp_frame_paths, process_frames, "face_processor")


def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue = Queue()
    for temp_frame_path in temp_frame_paths:
        queue.put(temp_frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    temp_frame_paths = []
    for _ in range(queue_per_future):
        if not queue.empty():
            temp_frame_paths.append(queue.get())
    return temp_frame_paths


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    # Formato más compacto para la barra de progreso
    progress_bar_format = '{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    total = len(frame_paths)
    
    with tqdm(total=total, desc='Processing frames', unit='frame', 
              dynamic_ncols=True, bar_format=progress_bar_format, 
              leave=False, position=0) as progress:
        
        # Usar procesamiento optimizado
        optimal_threads = get_optimal_thread_count()
        execution_threads = roop.globals.execution_threads if roop.globals.execution_threads is not None else optimal_threads
        
        with ThreadPoolExecutor(max_workers=execution_threads) as executor:
            futures = []
            queue = create_queue(frame_paths)
            queue_per_future = max(len(frame_paths) // execution_threads, 1)
            
            def update_with_frame(frame_index):
                update_progress(progress, "video_processor", frame_index + 1, total)
            
            while not queue.empty():
                batch_frames = pick_queue(queue, queue_per_future)
                future = executor.submit(process_frames, source_path, batch_frames, lambda: update_with_frame(len(futures)))
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[VIDEO_PROCESSOR] Error en procesamiento: {e}")


def update_progress(progress: Any = None, processor_name: str = "unknown", frame_number: int = 0, total_frames: int = 0) -> None:
    if progress:
        progress.update(1)
        progress.set_postfix({
            'frame': f'{frame_number}/{total_frames}',
            'processor': processor_name
        })
