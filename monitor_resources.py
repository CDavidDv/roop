#!/usr/bin/env python3
"""
Script para monitorear el uso de recursos durante el procesamiento
"""

import psutil
import time
import subprocess
import sys

def get_gpu_memory():
    """Obtener uso de memoria GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            return memory_allocated, memory_reserved
    except:
        pass
    return 0, 0

def get_ram_usage():
    """Obtener uso de RAM"""
    memory = psutil.virtual_memory()
    return memory.percent, memory.used / 1024**3

def monitor_resources():
    """Monitorear recursos en tiempo real"""
    print("📊 MONITOREANDO RECURSOS")
    print("=" * 50)
    
    while True:
        try:
            # RAM
            ram_percent, ram_gb = get_ram_usage()
            
            # GPU
            gpu_allocated, gpu_reserved = get_gpu_memory()
            
            # CPU
            cpu_percent = psutil.cpu_percent()
            
            print(f"🖥️ CPU: {cpu_percent:5.1f}% | 💾 RAM: {ram_gb:5.1f}GB ({ram_percent:5.1f}%) | 🎮 GPU: {gpu_allocated:5.2f}GB usado, {gpu_reserved:5.2f}GB reservado", end='\r')
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitoreo detenido")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

def main():
    """Función principal"""
    print("🚀 MONITOR DE RECURSOS")
    print("=" * 60)
    print("Presiona Ctrl+C para detener")
    print("=" * 60)
    
    monitor_resources()

if __name__ == "__main__":
    sys.exit(main()) 