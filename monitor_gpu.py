#!/usr/bin/env python3
import time
import psutil
import subprocess
import threading
from datetime import datetime

def get_gpu_info():
    """Obtener información de GPU usando nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        name = parts[0]
                        memory_used = int(parts[1])
                        memory_total = int(parts[2])
                        utilization = int(parts[3])
                        gpu_info.append({
                            'name': name,
                            'memory_used': memory_used,
                            'memory_total': memory_total,
                            'utilization': utilization
                        })
            return gpu_info
    except:
        pass
    return []

def get_ram_usage():
    """Obtener uso de RAM"""
    memory = psutil.virtual_memory()
    return {
        'used': memory.used / (1024**3),  # GB
        'total': memory.total / (1024**3),  # GB
        'percent': memory.percent
    }

def monitor_resources():
    """Monitorear recursos en tiempo real"""
    print("🔍 MONITOREO DE RECURSOS EN TIEMPO REAL")
    print("=" * 60)
    
    while True:
        try:
            # Información de GPU
            gpu_info = get_gpu_info()
            if gpu_info:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
                print("🎮 GPU:")
                for i, gpu in enumerate(gpu_info):
                    memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                    print(f"  GPU {i}: {gpu['name']}")
                    print(f"    VRAM: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({memory_percent:.1f}%)")
                    print(f"    Utilización: {gpu['utilization']}%")
            else:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - No se detectó GPU")
            
            # Información de RAM
            ram = get_ram_usage()
            print(f"🧠 RAM: {ram['used']:.1f}GB / {ram['total']:.1f}GB ({ram['percent']:.1f}%)")
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"💻 CPU: {cpu_percent:.1f}%")
            
            print("-" * 60)
            time.sleep(5)  # Actualizar cada 5 segundos
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoreo detenido por el usuario")
            break
        except Exception as e:
            print(f"❌ Error en monitoreo: {e}")
            time.sleep(5)

def check_gpu_providers():
    """Verificar proveedores de GPU disponibles"""
    print("🔍 VERIFICACIÓN DE PROVEEDORES GPU")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible para ONNX Runtime")
        else:
            print("❌ CUDA GPU no disponible para ONNX Runtime")
    except Exception as e:
        print(f"❌ Error ONNX Runtime: {e}")
    
    try:
        import torch
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"PyTorch VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPU devices: {len(gpus)}")
        if gpus:
            print(f"TensorFlow GPU: {gpus[0]}")
    except Exception as e:
        print(f"❌ Error TensorFlow: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            check_gpu_providers()
        elif sys.argv[1] == "--monitor":
            monitor_resources()
        else:
            print("Uso:")
            print("  python monitor_gpu.py --check    # Verificar proveedores GPU")
            print("  python monitor_gpu.py --monitor  # Monitoreo en tiempo real")
    else:
        print("🔍 MONITOREO DE RECURSOS")
        print("=" * 30)
        check_gpu_providers()
        print("\n" + "=" * 30)
        monitor_resources() 