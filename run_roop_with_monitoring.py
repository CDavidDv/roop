#!/usr/bin/env python3
import subprocess
import threading
import time
import sys
import os
from monitor_gpu_advanced import GPUMonitor

def run_roop_with_monitoring(source_path, target_path, output_path, additional_args=None):
    """
    Ejecutar roop con monitoreo de GPU en tiempo real
    """
    # Configurar argumentos básicos
    cmd = [
        'python', 'run.py',
        '-s', source_path,
        '-t', target_path,
        '-o', output_path,
        '--execution-provider', 'cuda',
        '--max-memory', '8',  # Limitar RAM a 8GB para optimizar VRAM
        '--execution-threads', '8',
        '--gpu-memory-wait', '10'  # Esperar 10s entre procesadores
    ]
    
    # Agregar argumentos adicionales si se proporcionan
    if additional_args:
        cmd.extend(additional_args)
    
    print("🚀 INICIANDO ROOP CON MONITOREO DE GPU")
    print("=" * 60)
    print(f"📁 Origen: {source_path}")
    print(f"🎯 Destino: {target_path}")
    print(f"💾 Salida: {output_path}")
    print("=" * 60)
    
    # Iniciar monitoreo de GPU
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        # Ejecutar roop
        print("▶️ Ejecutando roop...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitorear salida de roop
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[ROOP] {output.strip()}")
        
        # Esperar a que termine el proceso
        return_code = process.poll()
        
        if return_code == 0:
            print("✅ Procesamiento completado exitosamente!")
        else:
            print(f"❌ Procesamiento falló con código: {return_code}")
            
    except KeyboardInterrupt:
        print("\n🛑 Procesamiento interrumpido por el usuario")
        process.terminate()
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
    finally:
        # Detener monitoreo
        monitor.stop_monitoring()
        print("📊 Monitoreo detenido")

def main():
    if len(sys.argv) < 4:
        print("Uso: python run_roop_with_monitoring.py <imagen_origen> <video_destino> <archivo_salida> [args_adicionales...]")
        print("\nEjemplo:")
        print("  python run_roop_with_monitoring.py cara.jpg video.mp4 resultado.mp4")
        print("  python run_roop_with_monitoring.py cara.jpg video.mp4 resultado.mp4 --keep-fps --many-faces")
        return
    
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    output_path = sys.argv[3]
    additional_args = sys.argv[4:] if len(sys.argv) > 4 else None
    
    # Verificar que los archivos existan
    if not os.path.exists(source_path):
        print(f"❌ Error: No se encuentra el archivo de origen: {source_path}")
        return
    
    if not os.path.exists(target_path):
        print(f"❌ Error: No se encuentra el archivo de destino: {target_path}")
        return
    
    # Verificar recursos antes de empezar
    monitor = GPUMonitor()
    print("🔍 VERIFICACIÓN INICIAL DE RECURSOS")
    print("=" * 40)
    
    gpu_info = monitor.get_gpu_info()
    if gpu_info:
        print(f"✅ GPU: {gpu_info[0]['name']}")
        print(f"📊 VRAM disponible: {gpu_info[0]['memory_total_mb']/1024:.1f}GB")
    else:
        print("❌ No se detectó GPU - el procesamiento será más lento")
    
    ram = monitor.get_ram_usage()
    print(f"🧠 RAM disponible: {ram['total_gb']:.1f}GB")
    
    # Mostrar consejos de optimización
    print("\n💡 CONSEJOS DE OPTIMIZACIÓN:")
    for tip in monitor.get_optimization_tips():
        print(f"  {tip}")
    
    print("\n" + "=" * 60)
    
    # Confirmar antes de continuar
    response = input("¿Continuar con el procesamiento? (y/n): ").lower()
    if response != 'y':
        print("❌ Procesamiento cancelado")
        return
    
    # Ejecutar roop con monitoreo
    run_roop_with_monitoring(source_path, target_path, output_path, additional_args)

if __name__ == "__main__":
    main() 