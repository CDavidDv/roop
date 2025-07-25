#!/usr/bin/env python3

import os
import sys
import warnings

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

# Configuración optimizada para GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suprimir warnings de versiones nuevas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules, process_video_with_memory_management
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

# Suprimir warnings específicos de librerías
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, cuda, ...)', dest='execution_provider', default=['cuda'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('--gpu-memory-wait', help='wait time between processors to free GPU memory (seconds)', dest='gpu_memory_wait', type=int, default=15)
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads
    roop.globals.gpu_memory_wait_time = args.gpu_memory_wait


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available_providers = onnxruntime.get_available_providers()
    encoded_providers = encode_execution_providers(available_providers)
    
    # Filtrar proveedores solicitados que estén disponibles
    selected_providers = []
    for requested_provider in execution_providers:
        for provider, encoded_provider in zip(available_providers, encoded_providers):
            if requested_provider.lower() in encoded_provider.lower():
                selected_providers.append(provider)
                break
    
    # Si no se encontró ningún proveedor solicitado, usar el primero disponible
    if not selected_providers and available_providers:
        selected_providers = [available_providers[0]]
    
    return selected_providers


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    available_providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        # Con GPU, usar menos hilos para evitar saturación
        return 8
    elif 'ROCMExecutionProvider' in available_providers:
        # Con AMD GPU, usar configuración similar a CUDA
        return 8
    else:
        # Solo CPU, usar más hilos
        return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    try:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
                tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
            ])
    except Exception as e:
        print(f"⚠️ Warning: No se pudo configurar TensorFlow GPU: {e}")

    # Configurar ONNX Runtime para usar GPU de manera eficiente
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ ONNX Runtime CUDA disponible")
            # Configurar opciones de CUDA para mejor rendimiento
            cuda_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_use_max_workspace': '1',
                'do_copy_in_default_stream': '1',
            }
            print(f"🔧 Configuración CUDA: {cuda_options}")
        else:
            print("⚠️ ONNX Runtime CUDA no disponible")
    except Exception as e:
        print(f"⚠️ Warning: No se pudo configurar ONNX Runtime: {e}")


def pre_check() -> bool:
    if not shutil.which('ffmpeg'):
        print('ffmpeg is not installed!')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')


def start() -> None:
    for frame_processor_module in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor_module.pre_start():
            return
    limit_resources()
    if roop.globals.headless:
        if has_image_extension(roop.globals.target_path):
            if predict_image(roop.globals.target_path):
                process_image(roop.globals.source_path, roop.globals.target_path, roop.globals.output_path)
        elif is_video(roop.globals.target_path):
            if predict_video(roop.globals.target_path):
                process_video(roop.globals.source_path, roop.globals.target_path, roop.globals.output_path)
    else:
        ui.update_status('Select an image for source path.')
        ui.update_status('Select an image or video for target path.')
        ui.launch()


def destroy() -> None:
    for frame_processor_module in get_frame_processors_modules(roop.globals.frame_processors):
        frame_processor_module.post_process()
    clean_temp()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    if roop.globals.frame_processors:
        for frame_processor_module in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor_module.NAME)
            frame_processor_module.process_image(source_path, target_path, output_path)
            frame_processor_module.post_process()


def process_video(source_path: str, target_path: str, output_path: str) -> None:
    if roop.globals.frame_processors:
        temp_frame_paths = extract_frames(target_path, roop.globals.temp_frame_format, roop.globals.temp_frame_quality)
        if temp_frame_paths:
            for frame_processor_module in get_frame_processors_modules(roop.globals.frame_processors):
                update_status('Progressing...', frame_processor_module.NAME)
                frame_processor_module.process_video(source_path, temp_frame_paths)
                frame_processor_module.post_process()
            if roop.globals.keep_fps:
                fps = detect_fps(target_path)
                create_video(target_path, output_path, fps)
            else:
                create_video(target_path, output_path)
            restore_audio(target_path, output_path, roop.globals.skip_audio)
            move_temp(target_path, output_path, roop.globals.output_path)
            clean_temp(target_path)


def run() -> None:
    parse_args()
    if not pre_check():
        return
    start()
    destroy()


if __name__ == '__main__':
    run()
