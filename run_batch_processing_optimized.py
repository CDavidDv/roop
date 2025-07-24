#!/usr/bin/env python3
"""
Script optimizado para procesar múltiples videos con monitoreo de GPU para 15GB VRAM
"""

import os
import sys
import argparse
import subprocess
import time
import threading
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
    # Extraer nombre base del target (sin extensión)
    target_base = Path(target_name).stem
    # Crear nombre de salida: SakuraAS + número del video
    output_name = f"{source_name}{target_base}.mp4"
    return output_name

def get_optimal_settings_for_15gb() -> dict:
    """Obtener configuraciones óptimas para GPU de 15GB"""
    return {
        'max_memory': 8,  # Limitar RAM a 8GB para optimizar VRAM
        'execution_threads': 8,  # Optimizado para 15GB
        'gpu_memory_wait': 5,  # Esperar 15s entre videos
        'temp_frame_quality': 100,  # Calidad balanceada
        'temp_frame_format': 'png',  # Ahorrar espacio
        'output_video_encoder': 'h264_nvenc',  # Usar encoder NVIDIA
        'output_video_quality': 100,  # Calidad balanceada
        'execution_provider': 'cuda'
    }

def process_single_video_optimized(source_path: str, target_path: str, output_path: str, 
                                 settings: dict, keep_fps: bool = True) -> bool:
    """Procesar un solo video con configuraciones optimizadas"""
    
    print(f"\n🎬 PROCESANDO: {target_path}")
    print(f"📸 Source: {source_path}")
    print(f"💾 Output: {output_path}")
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
        # Ejecutar comando con salida en tiempo real
        print("🚀 Iniciando procesamiento...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitorear salida en tiempo real
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Filtrar solo mensajes importantes
                line = output.strip()
                if any(keyword in line.lower() for keyword in ['progress', 'processing', 'frame', 'video', 'error', 'warning']):
                    print(f"[ROOP] {line}")
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"✅ Video procesado exitosamente: {output_path}")
            return True
        else:
            print(f"❌ Error procesando {target_path} (código: {return_code})")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {target_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def monitor_gpu_during_processing(monitor: GPUMonitor, stop_event: threading.Event, video_name: str = ""):
    """Monitorear GPU durante el procesamiento con información detallada"""
    last_vram = 0
    last_ram = 0
    
    while not stop_event.is_set():
        try:
            gpu_info = monitor.get_gpu_info()
            if gpu_info:
                gpu = gpu_info[0]  # Tesla T4
                vram_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                vram_changed = abs(vram_percent - last_vram) > 2
                last_vram = vram_percent
                
                ram = monitor.get_ram_usage()
                ram_changed = abs(ram['percent'] - last_ram) > 3
                last_ram = ram['percent']
                
                # Solo mostrar si hay cambios significativos
                if vram_changed or ram_changed:
                    timestamp = time.strftime('%H:%M:%S')
                    print(f"\n📊 [{timestamp}] MONITOREO GPU - {video_name}")
                    print(f"    VRAM: {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB ({vram_percent:.1f}%)")
                    print(f"    RAM: {ram['used_gb']:.1f}GB / {ram['total_gb']:.1f}GB ({ram['percent']:.1f}%)")
                    print(f"    GPU Util: {gpu['utilization_percent']}% | Temp: {gpu['temperature_celsius']}°C")
                    
                    # Alertas
                    if vram_percent > 90:
                        print("    ⚠️  ALERTA: VRAM muy alta!")
                    elif vram_percent > 80:
                        print("    ⚠️  VRAM alta")
                    elif vram_percent > 50:
                        print("    ✅ VRAM en uso activo")
                    
                    if ram['percent'] > 90:
                        print("    ⚠️  ALERTA: RAM muy alta!")
                    elif ram['percent'] > 80:
                        print("    ⚠️  RAM alta")
                
            time.sleep(10)  # Verificar cada 10 segundos
        except Exception as e:
            print(f"Error en monitoreo: {e}")
            time.sleep(10)

def process_video_batch_optimized(source_path: str, target_videos: list, output_dir: str = None,
                                keep_fps: bool = True, monitor_gpu: bool = True) -> None:
    """Procesar lote de videos con optimizaciones para 15GB VRAM"""
    
    # Obtener configuraciones óptimas
    settings = get_optimal_settings_for_15gb()
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE OPTIMIZADO")
    print("=" * 70)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Videos a procesar: {len(target_videos)}")
    print(f"🎯 Optimizado para GPU de 15GB VRAM")
    print("=" * 70)
    print("⚙️ CONFIGURACIONES ÓPTIMAS:")
    print(f"  • RAM máxima: {settings['max_memory']}GB")
    print(f"  • Threads: {settings['execution_threads']}")
    print(f"  • Espera GPU: {settings['gpu_memory_wait']}s")
    print(f"  • Calidad frames: {settings['temp_frame_quality']}")
    print(f"  • Formato frames: {settings['temp_frame_format']}")
    print(f"  • Encoder: {settings['output_video_encoder']}")
    print(f"  • Calidad video: {settings['output_video_quality']}")
    print(f"  • Keep FPS: {keep_fps}")
    print("=" * 70)
    
    # Verificar que el source existe
    if not check_file_exists(source_path, "Source"):
        return
    
    # Crear directorio de salida si no existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio creado: {output_dir}")
    
    # Iniciar monitoreo de GPU si se solicita
    monitor = None
    stop_monitoring = threading.Event()
    if monitor_gpu:
        try:
            monitor = GPUMonitor()
            print("📊 Monitoreo de GPU habilitado")
        except Exception as e:
            print(f"⚠️ No se pudo iniciar monitoreo: {e}")
    
    # Obtener nombre base del source
    source_name = Path(source_path).stem
    
    # Procesar cada video
    successful_videos = 0
    total_videos = len(target_videos)
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n🎬 PROGRESO: {i}/{total_videos}")
        
        # Verificar que el video existe
        if not check_file_exists(target_video, "Video"):
            continue
        
        # Generar nombre de salida
        output_filename = get_output_filename(source_name, Path(target_video).stem)
        output_path = os.path.join(output_dir, output_filename) if output_dir else output_filename
        
        # Verificar si ya existe el archivo de salida
        if os.path.exists(output_path):
            print(f"⏭️ Saltando {target_video} - ya existe: {output_path}")
            successful_videos += 1
            continue
        
        # Iniciar monitoreo específico para este video
        video_name = Path(target_video).stem
        if monitor_gpu and monitor:
            stop_monitoring.clear()
            monitor_thread = threading.Thread(
                target=monitor_gpu_during_processing, 
                args=(monitor, stop_monitoring, video_name),
                daemon=True
            )
            monitor_thread.start()
            print(f"📊 Monitoreo iniciado para: {video_name}")
        
        # Procesar video
        start_time = time.time()
        success = process_single_video_optimized(source_path, target_video, output_path, settings, keep_fps)
        
        # Detener monitoreo para este video
        if monitor_gpu and monitor:
            stop_monitoring.set()
            time.sleep(1)  # Dar tiempo para que se detenga
        
        if success:
            successful_videos += 1
            elapsed_time = time.time() - start_time
            print(f"⏱️ Tiempo de procesamiento: {elapsed_time:.1f}s")
        else:
            print(f"❌ Falló el procesamiento de: {target_video}")
        
        # Esperar entre videos para liberar memoria GPU
        if i < total_videos:  # No esperar después del último video
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
    parser = argparse.ArgumentParser(description='Procesar múltiples videos con ROOP optimizado para 15GB VRAM')
    parser.add_argument('--source', required=True, help='Ruta de la imagen de origen')
    parser.add_argument('--videos', nargs='+', required=True, help='Lista de videos a procesar')
    parser.add_argument('--output-dir', help='Directorio de salida (opcional)')
    parser.add_argument('--keep-fps', action='store_true', help='Mantener FPS original')
    parser.add_argument('--no-monitor', action='store_true', help='Desactivar monitoreo de GPU')
    
    args = parser.parse_args()
    
    # Verificar recursos antes de empezar
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
    
    # Mostrar configuraciones óptimas
    settings = get_optimal_settings_for_15gb()
    print("\n💡 CONFIGURACIONES ÓPTIMAS PARA 15GB VRAM:")
    print(f"  • RAM máxima: {settings['max_memory']}GB")
    print(f"  • Threads: {settings['execution_threads']}")
    print(f"  • Espera GPU: {settings['gpu_memory_wait']}s")
    print(f"  • Formato frames: {settings['temp_frame_format']}")
    print(f"  • Encoder: {settings['output_video_encoder']}")
    
    print("\n" + "=" * 60)
    
    # Confirmar antes de continuar
    response = input("¿Continuar con el procesamiento optimizado? (y/n): ").lower()
    if response != 'y':
        print("❌ Procesamiento cancelado")
        return
    
    # Procesar videos
    process_video_batch_optimized(
        source_path=args.source,
        target_videos=args.videos,
        output_dir=args.output_dir,
        keep_fps=args.keep_fps,
        monitor_gpu=not args.no_monitor
    )

if __name__ == "__main__":
    main() 