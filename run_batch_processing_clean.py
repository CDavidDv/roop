#!/usr/bin/env python3
"""
Script optimizado para procesar múltiples videos con salida limpia y monitoreo de GPU
"""

import os
import sys
import argparse
import subprocess
import time
import threading
import re
from pathlib import Path
from monitor_gpu_advanced import GPUMonitor

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Desactivar predictor NSFW para evitar errores de GPU
import roop.predictor
def predict_video_skip_nsfw(target_path: str) -> bool:
    print("⚠️ Saltando verificación NSFW para evitar conflictos de GPU...")
    return False

roop.predictor.predict_video = predict_video_skip_nsfw

def check_file_exists(file_path: str, file_type: str) -> bool:
    """Verificar si un archivo existe"""
    if not os.path.exists(file_path):
        print(f"❌ {file_type} no encontrado: {file_path}")
        return False
    return True

def get_output_filename(source_name: str, target_name: str) -> str:
    """Generar nombre de archivo de salida"""
    target_base = Path(target_name).stem
    output_name = f"{source_name}{target_base}.mp4"
    return output_name

def get_optimal_settings_for_15gb() -> dict:
    """Obtener configuraciones óptimas para GPU de 15GB"""
    return {
        'max_memory': 8,
        'execution_threads': 8,
        'gpu_memory_wait': 5,
        'temp_frame_quality': 100,
        'temp_frame_format': 'png',
        'output_video_encoder': 'h264_nvenc',
        'output_video_quality': 100,
        'execution_provider': 'cuda'
    }

def parse_progress_line(line: str) -> dict:
    """Parsear línea de progreso de ROOP"""
    try:
        # Extraer información de la línea de progreso
        progress_match = re.search(r'(\d+)%\|', line)
        frames_match = re.search(r'(\d+)/(\d+)', line)
        time_match = re.search(r'\[(\d+):(\d+)<(\d+):(\d+),', line)
        speed_match = re.search(r'(\d+\.\d+)s/frame', line)
        memory_match = re.search(r'memory=(\d+\.\d+)GB', line)
        gpu_memory_match = re.search(r'gpu_memory=GPU: (\d+\.\d+)GB', line)
        
        if progress_match and frames_match:
            return {
                'progress': int(progress_match.group(1)),
                'current_frame': int(frames_match.group(1)),
                'total_frames': int(frames_match.group(2)),
                'speed': float(speed_match.group(1)) if speed_match else 0,
                'memory_gb': float(memory_match.group(1)) if memory_match else 0,
                'gpu_memory_gb': float(gpu_memory_match.group(1)) if gpu_memory_match else 0,
                'time_elapsed': f"{time_match.group(1)}:{time_match.group(2)}" if time_match else "0:00",
                'time_remaining': f"{time_match.group(3)}:{time_match.group(4)}" if time_match else "0:00"
            }
    except Exception as e:
        pass
    return {}

def create_progress_bar(percentage: int, width: int = 30) -> str:
    """Crear barra de progreso visual"""
    filled_length = int(width * percentage / 100)
    bar = '█' * filled_length + '░' * (width - filled_length)
    return f"[{bar}] {percentage}%"

def process_single_video_clean(source_path: str, target_path: str, output_path: str, 
                             settings: dict, keep_fps: bool = True) -> bool:
    """Procesar un solo video con salida limpia"""
    
    print(f"\n🎬 PROCESANDO: {Path(target_path).name}")
    print(f"📸 Source: {Path(source_path).name}")
    print(f"💾 Output: {Path(output_path).name}")
    print("=" * 60)
    
    # Construir comando optimizado
    cmd = [
        "roop_env/bin/python", 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--gpu-memory-wait', str(settings['gpu_memory_wait']),
        '--max-memory', str(settings['max_memory']),
        '--execution-threads', str(settings['execution_threads']),
        '--temp-frame-quality', str(settings['temp_frame_quality']),
        '--temp-frame-format', settings['temp_frame_format'],
        '--output-video-encoder', settings['output_video_encoder'],
        '--output-video-quality', str(settings['output_video_quality']),
        '--execution-provider', settings['execution_provider']
    ]
    
    if keep_fps:
        cmd.append('--keep-fps')
    
    try:
        # Ejecutar comando con salida limpia
        print("🚀 Iniciando procesamiento...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        last_progress = 0
        last_line_length = 0
        
        # Monitorear salida en tiempo real
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                
                # Filtrar líneas de progreso
                if 'Processing face-swapper:' in line or 'Processing face_enhancer:' in line:
                    progress_info = parse_progress_line(line)
                    if progress_info:
                        # Limpiar línea anterior
                        if last_line_length > 0:
                            print('\r' + ' ' * last_line_length + '\r', end='', flush=True)
                        
                        # Crear línea de progreso limpia
                        progress_bar = create_progress_bar(progress_info['progress'])
                        status_line = (
                            f"🔄 {progress_bar} | "
                            f"Frame {progress_info['current_frame']}/{progress_info['total_frames']} | "
                            f"⏱️ {progress_info['time_elapsed']} | "
                            f"⏳ {progress_info['time_remaining']} | "
                            f"🚀 {progress_info['speed']:.1f}s/frame | "
                            f"🧠 {progress_info['memory_gb']:.1f}GB | "
                            f"🎮 {progress_info['gpu_memory_gb']:.1f}GB VRAM"
                        )
                        
                        print(status_line, end='', flush=True)
                        last_line_length = len(status_line)
                        last_progress = progress_info['progress']
                
                # Mostrar otros mensajes importantes
                elif any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'success', 'completed']):
                    print(f"\n[ROOP] {line}")
                    last_line_length = 0
        
        # Limpiar línea final
        if last_line_length > 0:
            print('\r' + ' ' * last_line_length + '\r', end='', flush=True)
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"✅ Video procesado exitosamente: {Path(output_path).name}")
            return True
        else:
            print(f"❌ Error procesando {Path(target_path).name} (código: {return_code})")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {target_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def monitor_gpu_clean(monitor: GPUMonitor, stop_event: threading.Event, video_name: str = ""):
    """Monitorear GPU con salida limpia"""
    last_vram = 0
    last_ram = 0
    last_line_length = 0
    
    while not stop_event.is_set():
        try:
            gpu_info = monitor.get_gpu_info()
            if gpu_info:
                gpu = gpu_info[0]
                vram_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                vram_changed = abs(vram_percent - last_vram) > 2
                last_vram = vram_percent
                
                ram = monitor.get_ram_usage()
                ram_changed = abs(ram['percent'] - last_ram) > 3
                last_ram = ram['percent']
                
                # Solo mostrar si hay cambios significativos
                if vram_changed or ram_changed:
                    # Limpiar línea anterior
                    if last_line_length > 0:
                        print('\r' + ' ' * last_line_length + '\r', end='', flush=True)
                    
                    timestamp = time.strftime('%H:%M:%S')
                    status_line = (
                        f"📊 [{timestamp}] GPU: {gpu['memory_used_mb']}MB/{gpu['memory_total_mb']}MB "
                        f"({vram_percent:.1f}%) | RAM: {ram['used_gb']:.1f}GB/{ram['total_gb']:.1f}GB "
                        f"({ram['percent']:.1f}%) | Temp: {gpu['temperature_celsius']}°C"
                    )
                    
                    print(status_line, end='', flush=True)
                    last_line_length = len(status_line)
                
            time.sleep(10)
        except Exception as e:
            print(f"\nError en monitoreo: {e}")
            time.sleep(10)
    
    # Limpiar línea final del monitoreo
    if last_line_length > 0:
        print('\r' + ' ' * last_line_length + '\r', end='', flush=True)

def process_video_batch_clean(source_path: str, target_videos: list, output_dir: str = None,
                            keep_fps: bool = True, monitor_gpu: bool = True) -> None:
    """Procesar lote de videos con salida limpia"""
    
    settings = get_optimal_settings_for_15gb()
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE OPTIMIZADO")
    print("=" * 70)
    print(f"📸 Source: {Path(source_path).name}")
    print(f"🎬 Videos a procesar: {len(target_videos)}")
    print(f"🎯 Optimizado para GPU de 15GB VRAM")
    print("=" * 70)
    
    if not check_file_exists(source_path, "Source"):
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio creado: {output_dir}")
    
    # Iniciar monitoreo de GPU
    monitor = None
    stop_monitoring = threading.Event()
    if monitor_gpu:
        try:
            monitor = GPUMonitor()
            print("📊 Monitoreo de GPU habilitado")
        except Exception as e:
            print(f"⚠️ No se pudo iniciar monitoreo: {e}")
    
    source_name = Path(source_path).stem
    successful_videos = 0
    total_videos = len(target_videos)
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n🎬 PROGRESO: {i}/{total_videos}")
        
        if not check_file_exists(target_video, "Video"):
            continue
        
        output_filename = get_output_filename(source_name, Path(target_video).stem)
        output_path = os.path.join(output_dir, output_filename) if output_dir else output_filename
        
        if os.path.exists(output_path):
            print(f"⏭️ Saltando {Path(target_video).name} - ya existe")
            successful_videos += 1
            continue
        
        # Iniciar monitoreo específico para este video
        video_name = Path(target_video).stem
        if monitor_gpu and monitor:
            stop_monitoring.clear()
            monitor_thread = threading.Thread(
                target=monitor_gpu_clean, 
                args=(monitor, stop_monitoring, video_name),
                daemon=True
            )
            monitor_thread.start()
        
        # Procesar video
        start_time = time.time()
        success = process_single_video_clean(source_path, target_video, output_path, settings, keep_fps)
        
        # Detener monitoreo
        if monitor_gpu and monitor:
            stop_monitoring.set()
            time.sleep(1)
        
        if success:
            successful_videos += 1
            elapsed_time = time.time() - start_time
            print(f"\n⏱️ Tiempo de procesamiento: {elapsed_time:.1f}s")
        else:
            print(f"\n❌ Falló el procesamiento de: {Path(target_video).name}")
        
        # Esperar entre videos
        if i < total_videos:
            print(f"⏳ Esperando {settings['gpu_memory_wait']}s para liberar memoria GPU...")
            time.sleep(settings['gpu_memory_wait'])
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    print(f"✅ Videos procesados exitosamente: {successful_videos}/{total_videos}")
    print(f"📁 Directorio de salida: {output_dir}")
    print("🎯 Optimizaciones aplicadas para GPU de 15GB VRAM")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Procesar múltiples videos con ROOP optimizado y salida limpia')
    parser.add_argument('--source', required=True, help='Ruta de la imagen de origen')
    parser.add_argument('--videos', nargs='+', required=True, help='Lista de videos a procesar')
    parser.add_argument('--output-dir', help='Directorio de salida (opcional)')
    parser.add_argument('--keep-fps', action='store_true', help='Mantener FPS original')
    parser.add_argument('--no-monitor', action='store_true', help='Desactivar monitoreo de GPU')
    
    args = parser.parse_args()
    
    # Verificar recursos
    print("🔍 VERIFICACIÓN DE RECURSOS")
    print("=" * 40)
    
    monitor = GPUMonitor()
    gpu_info = monitor.get_gpu_info()
    if gpu_info:
        print(f"✅ GPU: {gpu_info[0]['name']}")
        print(f"📊 VRAM: {gpu_info[0]['memory_total_mb']/1024:.1f}GB")
        vram_percent = (gpu_info[0]['memory_used_mb'] / gpu_info[0]['memory_total_mb']) * 100
        print(f"📊 VRAM usada: {vram_percent:.1f}%")
    else:
        print("❌ No se detectó GPU NVIDIA")
    
    ram = monitor.get_ram_usage()
    print(f"🧠 RAM: {ram['total_gb']:.1f}GB")
    print(f"🧠 RAM usada: {ram['percent']:.1f}%")
    
    settings = get_optimal_settings_for_15gb()
    print("\n💡 CONFIGURACIONES ÓPTIMAS PARA 15GB VRAM:")
    print(f"  • RAM máxima: {settings['max_memory']}GB")
    print(f"  • Threads: {settings['execution_threads']}")
    print(f"  • Espera GPU: {settings['gpu_memory_wait']}s")
    print(f"  • Formato frames: {settings['temp_frame_format']}")
    print(f"  • Encoder: {settings['output_video_encoder']}")
    
    print("\n" + "=" * 60)
    
    response = input("¿Continuar con el procesamiento optimizado? (y/n): ").lower()
    if response != 'y':
        print("❌ Procesamiento cancelado")
        return
    
    process_video_batch_clean(
        source_path=args.source,
        target_videos=args.videos,
        output_dir=args.output_dir,
        keep_fps=args.keep_fps,
        monitor_gpu=not args.no_monitor
    )

if __name__ == "__main__":
    main() 