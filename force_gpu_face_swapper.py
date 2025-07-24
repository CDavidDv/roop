#!/usr/bin/env python3
"""
Script para forzar el uso de GPU en face-swapper
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def backup_original_file():
    """Hacer backup del archivo original"""
    face_swapper_path = Path("roop/processors/frame/face_swapper.py")
    backup_path = Path("roop/processors/frame/face_swapper_backup.py")
    
    if face_swapper_path.exists():
        shutil.copy2(face_swapper_path, backup_path)
        print(f"✅ Backup creado: {backup_path}")
        return True
    else:
        print(f"❌ Archivo no encontrado: {face_swapper_path}")
        return False

def create_gpu_forced_face_swapper():
    """Crear versión del face-swapper que fuerza GPU"""
    
    gpu_forced_code = '''from typing import Any, List, Callable
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
NAME = 'ROOP.FACE-SWAPPER-GPU'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../inswapper_128.onnx')
            
            # FORZAR USO DE GPU
            print(f"[{NAME}] 🔧 CONFIGURANDO FORZADO DE GPU...")
            
            # Configurar variables de entorno para GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Verificar proveedores disponibles
            available_providers = ort.get_available_providers()
            print(f"[{NAME}] Proveedores disponibles: {available_providers}")
            
            # FORZAR SOLO CUDA
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider']
                print(f"[{NAME}] ✅ FORZANDO USO DE GPU (CUDA)")
                print(f"[{NAME}] Cargando modelo con proveedores: {providers}")
                
                # Configurar opciones específicas de CUDA
                cuda_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_use_max_workspace': '1',
                    'do_copy_in_default_stream': '1',
                }
                
                # Crear sesión con configuración específica
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                
                print(f"[{NAME}] Configurando sesión ONNX con opciones CUDA...")
                
                # Cargar modelo con configuración forzada
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path, 
                    providers=providers,
                    session_options=session_options
                )
                
                print(f"[{NAME}] ✅ Modelo cargado con GPU forzado")
                
            else:
                print(f"[{NAME}] ❌ CUDA no disponible, usando CPU")
                providers = ['CPUExecutionProvider']
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
            
            # Verificar proveedores aplicados
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
    process_frames(source_path, temp_frame_paths, lambda: update_status())
'''
    
    return gpu_forced_code

def apply_gpu_forced_face_swapper():
    """Aplicar la versión forzada de GPU al face-swapper"""
    
    face_swapper_path = Path("roop/processors/frame/face_swapper.py")
    
    if not face_swapper_path.exists():
        print(f"❌ Archivo no encontrado: {face_swapper_path}")
        return False
    
    # Crear backup
    if not backup_original_file():
        return False
    
    # Crear código forzado
    gpu_forced_code = create_gpu_forced_face_swapper()
    
    # Escribir nuevo archivo
    try:
        with open(face_swapper_path, 'w') as f:
            f.write(gpu_forced_code)
        
        print(f"✅ Face-swapper modificado para forzar GPU")
        print(f"📁 Archivo: {face_swapper_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error modificando archivo: {e}")
        return False

def test_gpu_usage():
    """Probar el uso de GPU después de la modificación"""
    
    print("\n🧪 PROBANDO USO DE GPU")
    print("=" * 40)
    
    # Comando de prueba
    test_cmd = [
        "roop_env/bin/python", 'run.py',
        '--source', '/content/DanielaAS.jpg',
        '--target', '/content/112.mp4',
        '-o', '/content/test_gpu_face_swapper.mp4',
        '--frame-processor', 'face_swapper',
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '5',
        '--temp-frame-quality', '100',
        '--temp-frame-format', 'png',
        '--output-video-encoder', 'h264_nvenc',
        '--output-video-quality', '100',
        '--keep-fps'
    ]
    
    print("Comando de prueba:")
    print(" ".join(test_cmd))
    
    print("\n💡 Para probar:")
    print("1. Ejecuta el comando anterior")
    print("2. Monitorea con: nvidia-smi -l 1")
    print("3. Verifica que VRAM > 0GB durante el procesamiento")
    print("4. Busca mensajes '[ROOP.FACE-SWAPPER-GPU]' en la salida")

def restore_original_file():
    """Restaurar archivo original"""
    
    face_swapper_path = Path("roop/processors/frame/face_swapper.py")
    backup_path = Path("roop/processors/frame/face_swapper_backup.py")
    
    if backup_path.exists():
        shutil.copy2(backup_path, face_swapper_path)
        print(f"✅ Archivo original restaurado: {face_swapper_path}")
        return True
    else:
        print(f"❌ Backup no encontrado: {backup_path}")
        return False

def main():
    print("🔧 FORZADOR DE GPU PARA FACE-SWAPPER")
    print("=" * 50)
    
    print("1. Crear backup del archivo original")
    print("2. Modificar face-swapper para forzar GPU")
    print("3. Probar uso de GPU")
    print("4. Restaurar archivo original (opcional)")
    print("=" * 50)
    
    # Aplicar modificación
    if apply_gpu_forced_face_swapper():
        print("\n✅ MODIFICACIÓN APLICADA")
        print("=" * 30)
        print("El face-swapper ahora:")
        print("• Fuerza el uso de CUDA GPU")
        print("• Configura variables de entorno para GPU")
        print("• Usa configuración específica de CUDA")
        print("• Muestra mensajes de diagnóstico")
        
        # Probar
        test_gpu_usage()
        
        print("\n" + "=" * 50)
        print("🔄 RESTAURAR ARCHIVO ORIGINAL:")
        print("=" * 50)
        response = input("¿Restaurar archivo original después de probar? (y/n): ").lower()
        if response == 'y':
            restore_original_file()
        else:
            print("💡 Para restaurar manualmente: python force_gpu_face_swapper.py --restore")
    else:
        print("❌ No se pudo aplicar la modificación")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        restore_original_file()
    else:
        main() 