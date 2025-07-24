#!/usr/bin/env python3
import time
import psutil
import subprocess
import threading
import os
from datetime import datetime
from typing import Dict, List

class LiveGPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.last_vram = 0
        self.last_ram = 0
        self.last_cpu = 0
        
    def get_gpu_info(self) -> List[Dict]:
        """Obtener información detallada de GPU"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 6:
                            gpu_info.append({
                                'name': parts[0],
                                'memory_used_mb': int(parts[1]),
                                'memory_total_mb': int(parts[2]),
                                'utilization_percent': int(parts[3]),
                                'temperature_celsius': int(parts[4]),
                                'power_watts': float(parts[5]) if parts[5] != 'N/A' else 0
                            })
                return gpu_info
        except Exception as e:
            print(f"Error obteniendo info GPU: {e}")
        return []

    def get_ram_usage(self) -> Dict:
        """Obtener uso de RAM"""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        }

    def get_process_gpu_usage(self) -> Dict:
        """Obtener procesos usando GPU"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                processes = {}
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            pid = int(parts[0])
                            process_name = parts[1]
                            memory_mb = int(parts[2])
                            processes[pid] = {
                                'name': process_name,
                                'memory_mb': memory_mb
                            }
                return processes
        except:
            pass
        return {}

    def create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Crear barra de progreso visual"""
        filled_length = int(width * percentage / 100)
        bar = '█' * filled_length + '░' * (width - filled_length)
        return f"[{bar}] {percentage:.1f}%"

    def monitor_live(self):
        """Monitoreo en tiempo real con actualizaciones visibles"""
        print("🔍 MONITOREO EN TIEMPO REAL - GPU TESLA T4")
        print("=" * 70)
        print("⏰ Actualizaciones cada 5 segundos...")
        print("=" * 70)
        
        while self.monitoring:
            try:
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # Información de GPU
                gpu_info = self.get_gpu_info()
                if gpu_info:
                    gpu = gpu_info[0]  # Tesla T4
                    memory_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                    
                    # Detectar cambios significativos
                    vram_changed = abs(memory_percent - self.last_vram) > 1
                    self.last_vram = memory_percent
                    
                    print(f"\n⏰ {timestamp}")
                    print("🎮 TESLA T4:")
                    print(f"    VRAM: {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB")
                    print(f"    {self.create_progress_bar(memory_percent)}")
                    print(f"    Utilización: {gpu['utilization_percent']}%")
                    print(f"    Temperatura: {gpu['temperature_celsius']}°C")
                    print(f"    Potencia: {gpu['power_watts']:.1f}W")
                    
                    # Alertas específicas
                    if memory_percent > 90:
                        print("    ⚠️  ALERTA: VRAM muy alta!")
                    elif memory_percent > 80:
                        print("    ⚠️  VRAM alta - monitorea")
                    elif memory_percent > 50:
                        print("    ✅ VRAM en uso activo")
                    else:
                        print("    💤 VRAM en reposo")
                        
                    # Mostrar cambios
                    if vram_changed:
                        print(f"    📈 Cambio detectado en VRAM")
                else:
                    print(f"\n⏰ {timestamp} - No se detectó GPU")
                
                # Información de RAM
                ram = self.get_ram_usage()
                ram_changed = abs(ram['percent'] - self.last_ram) > 2
                self.last_ram = ram['percent']
                
                print(f"🧠 RAM: {ram['used_gb']:.1f}GB / {ram['total_gb']:.1f}GB")
                print(f"    {self.create_progress_bar(ram['percent'])}")
                
                if ram['percent'] > 90:
                    print("    ⚠️  ALERTA: RAM muy alta!")
                elif ram['percent'] > 80:
                    print("    ⚠️  RAM alta")
                
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_changed = abs(cpu_percent - self.last_cpu) > 5
                self.last_cpu = cpu_percent
                
                print(f"💻 CPU: {cpu_percent:.1f}%")
                print(f"    {self.create_progress_bar(cpu_percent)}")
                
                # Procesos usando GPU
                gpu_processes = self.get_process_gpu_usage()
                if gpu_processes:
                    print("🔍 Procesos usando GPU:")
                    total_gpu_memory = 0
                    for pid, process in gpu_processes.items():
                        print(f"    PID {pid}: {process['name']} - {process['memory_mb']}MB")
                        total_gpu_memory += process['memory_mb']
                    print(f"    📊 Total procesos GPU: {total_gpu_memory}MB")
                else:
                    print("🔍 No hay procesos usando GPU")
                
                # Estado del procesamiento
                if memory_percent > 10 or cpu_percent > 20:
                    print("🔄 PROCESAMIENTO ACTIVO")
                else:
                    print("⏸️  PROCESAMIENTO EN PAUSA")
                
                print("-" * 70)
                time.sleep(5)  # Actualizar cada 5 segundos
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoreo detenido por el usuario")
                break
            except Exception as e:
                print(f"❌ Error en monitoreo: {e}")
                time.sleep(5)

    def start_monitoring(self):
        """Iniciar monitoreo"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_live()

def main():
    print("🚀 INICIANDO MONITOREO EN TIEMPO REAL")
    print("=" * 50)
    
    monitor = LiveGPUMonitor()
    
    # Verificación inicial
    print("🔍 VERIFICACIÓN INICIAL:")
    gpu_info = monitor.get_gpu_info()
    if gpu_info:
        gpu = gpu_info[0]
        print(f"✅ GPU detectada: {gpu['name']}")
        print(f"📊 VRAM total: {gpu['memory_total_mb']/1024:.1f}GB")
        print(f"📊 VRAM libre: {gpu['memory_total_mb'] - gpu['memory_used_mb']}MB")
    else:
        print("❌ No se detectó GPU NVIDIA")
    
    ram = monitor.get_ram_usage()
    print(f"🧠 RAM total: {ram['total_gb']:.1f}GB")
    print(f"🧠 RAM disponible: {ram['available_gb']:.1f}GB")
    
    print("\n" + "=" * 50)
    print("💡 CONSEJOS:")
    print("  • Si VRAM > 90%: Considera pausar el procesamiento")
    print("  • Si RAM > 90%: Cierra otras aplicaciones")
    print("  • Monitorea la temperatura (< 80°C)")
    print("  • Presiona Ctrl+C para detener")
    print("=" * 50)
    
    # Confirmar inicio
    response = input("\n¿Iniciar monitoreo en tiempo real? (y/n): ").lower()
    if response == 'y':
        monitor.start_monitoring()
    else:
        print("❌ Monitoreo cancelado")

if __name__ == "__main__":
    main() 