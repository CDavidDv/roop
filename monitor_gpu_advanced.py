#!/usr/bin/env python3
import time
import psutil
import subprocess
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

class GPUMonitor:
    def __init__(self, log_file: str = "gpu_monitor.log"):
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.gpu_history = []
        self.max_vram_gb = 15  # Tu GPU tiene 15GB
        
    def get_gpu_info(self) -> List[Dict]:
        """Obtener información detallada de GPU usando nvidia-smi"""
        try:
            # Información básica de GPU
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
        """Obtener uso detallado de RAM"""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        }

    def get_process_gpu_usage(self) -> Dict:
        """Obtener uso de GPU por proceso"""
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

    def log_metrics(self, metrics: Dict):
        """Guardar métricas en archivo de log"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Mantener solo las últimas 1000 entradas
        self.gpu_history.append(log_entry)
        if len(self.gpu_history) > 1000:
            self.gpu_history.pop(0)

    def get_vram_usage_percentage(self) -> float:
        """Obtener porcentaje de uso de VRAM"""
        gpu_info = self.get_gpu_info()
        if gpu_info:
            total_used = sum(gpu['memory_used_mb'] for gpu in gpu_info)
            total_available = sum(gpu['memory_total_mb'] for gpu in gpu_info)
            return (total_used / total_available) * 100
        return 0.0

    def check_vram_optimization(self) -> Dict:
        """Verificar si se puede optimizar el uso de VRAM"""
        vram_percent = self.get_vram_usage_percentage()
        ram = self.get_ram_usage()
        
        recommendations = []
        
        if vram_percent > 90:
            recommendations.append("⚠️ VRAM muy alta (>90%). Considera reducir batch size o procesar menos frames simultáneamente")
        elif vram_percent > 80:
            recommendations.append("⚠️ VRAM alta (>80%). Monitorea el uso")
        
        if ram['percent'] > 90:
            recommendations.append("⚠️ RAM muy alta (>90%). Considera cerrar otras aplicaciones")
        elif ram['percent'] > 80:
            recommendations.append("⚠️ RAM alta (>80%). Monitorea el uso")
        
        if vram_percent < 50 and ram['percent'] < 70:
            recommendations.append("✅ Recursos bien balanceados. Puedes aumentar batch size si es necesario")
        
        return {
            'vram_percent': vram_percent,
            'ram_percent': ram['percent'],
            'recommendations': recommendations
        }

    def monitor_resources(self):
        """Monitoreo en tiempo real con recomendaciones"""
        print("🔍 MONITOREO AVANZADO DE RECURSOS")
        print("=" * 70)
        print(f"🎯 Objetivo: Optimizar uso de {self.max_vram_gb}GB VRAM")
        print("=" * 70)
        
        while self.monitoring:
            try:
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # Información de GPU
                gpu_info = self.get_gpu_info()
                if gpu_info:
                    print(f"\n⏰ {timestamp}")
                    print("🎮 GPU:")
                    total_vram_used = 0
                    total_vram_total = 0
                    
                    for i, gpu in enumerate(gpu_info):
                        memory_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                        total_vram_used += gpu['memory_used_mb']
                        total_vram_total += gpu['memory_total_mb']
                        
                        print(f"  GPU {i}: {gpu['name']}")
                        print(f"    VRAM: {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB ({memory_percent:.1f}%)")
                        print(f"    Utilización: {gpu['utilization_percent']}%")
                        print(f"    Temperatura: {gpu['temperature_celsius']}°C")
                        print(f"    Potencia: {gpu['power_watts']:.1f}W")
                        
                        # Barra de progreso visual
                        bar_length = 20
                        filled_length = int(bar_length * memory_percent / 100)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f"    [{bar}] {memory_percent:.1f}%")
                else:
                    print(f"\n⏰ {timestamp} - No se detectó GPU")
                
                # Información de RAM
                ram = self.get_ram_usage()
                ram_percent = ram['percent']
                ram_bar_length = 20
                ram_filled = int(ram_bar_length * ram_percent / 100)
                ram_bar = '█' * ram_filled + '░' * (ram_bar_length - ram_filled)
                
                print(f"🧠 RAM: {ram['used_gb']:.1f}GB / {ram['total_gb']:.1f}GB ({ram_percent:.1f}%)")
                print(f"    [{ram_bar}] {ram_percent:.1f}%")
                
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_bar_length = 20
                cpu_filled = int(cpu_bar_length * cpu_percent / 100)
                cpu_bar = '█' * cpu_filled + '░' * (cpu_bar_length - cpu_filled)
                print(f"💻 CPU: {cpu_percent:.1f}%")
                print(f"    [{cpu_bar}] {cpu_percent:.1f}%")
                
                # Procesos usando GPU
                gpu_processes = self.get_process_gpu_usage()
                if gpu_processes:
                    print("🔍 Procesos usando GPU:")
                    for pid, process in gpu_processes.items():
                        print(f"  PID {pid}: {process['name']} - {process['memory_mb']}MB")
                
                # Recomendaciones
                optimization = self.check_vram_optimization()
                if optimization['recommendations']:
                    print("\n💡 RECOMENDACIONES:")
                    for rec in optimization['recommendations']:
                        print(f"  {rec}")
                
                # Guardar métricas
                metrics = {
                    'gpu_info': gpu_info,
                    'ram': ram,
                    'cpu_percent': cpu_percent,
                    'gpu_processes': gpu_processes,
                    'optimization': optimization
                }
                self.log_metrics(metrics)
                
                print("-" * 70)
                time.sleep(3)  # Actualizar cada 3 segundos
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoreo detenido por el usuario")
                break
            except Exception as e:
                print(f"❌ Error en monitoreo: {e}")
                time.sleep(5)

    def start_monitoring(self):
        """Iniciar monitoreo en hilo separado"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
            self.monitor_thread.start()
            print("✅ Monitoreo iniciado en segundo plano")

    def stop_monitoring(self):
        """Detener monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🛑 Monitoreo detenido")

    def get_optimization_tips(self) -> List[str]:
        """Obtener consejos de optimización para roop"""
        return [
            "🎯 Para 15GB VRAM - Configuraciones recomendadas:",
            "  • --max-memory 8 (limitar RAM a 8GB)",
            "  • --execution-provider cuda (usar GPU)",
            "  • --execution-threads 8 (optimizar threads)",
            "  • --gpu-memory-wait 10 (esperar 10s entre procesadores)",
            "",
            "📊 Monitoreo durante procesamiento:",
            "  • Ejecuta: python monitor_gpu_advanced.py --monitor",
            "  • Observa el uso de VRAM y RAM",
            "  • Si VRAM > 90%, reduce batch size",
            "  • Si RAM > 90%, cierra otras aplicaciones",
            "",
            "⚡ Optimizaciones adicionales:",
            "  • Usa --temp-frame-format jpg para ahorrar espacio",
            "  • Ajusta --output-video-quality según necesites",
            "  • Considera --skip-audio si no es necesario"
        ]

def main():
    monitor = GPUMonitor()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            print("🔍 VERIFICACIÓN DE RECURSOS")
            print("=" * 40)
            gpu_info = monitor.get_gpu_info()
            if gpu_info:
                print(f"✅ GPU detectada: {gpu_info[0]['name']}")
                print(f"📊 VRAM total: {gpu_info[0]['memory_total_mb']}MB ({gpu_info[0]['memory_total_mb']/1024:.1f}GB)")
            else:
                print("❌ No se detectó GPU")
            
            ram = monitor.get_ram_usage()
            print(f"🧠 RAM total: {ram['total_gb']:.1f}GB")
            
        elif sys.argv[1] == "--monitor":
            monitor.monitor_resources()
        elif sys.argv[1] == "--tips":
            print("\n".join(monitor.get_optimization_tips()))
        else:
            print("Uso:")
            print("  python monitor_gpu_advanced.py --check    # Verificar recursos")
            print("  python monitor_gpu_advanced.py --monitor  # Monitoreo en tiempo real")
            print("  python monitor_gpu_advanced.py --tips     # Consejos de optimización")
    else:
        print("🔍 MONITOREO AVANZADO DE GPU")
        print("=" * 40)
        monitor.monitor_resources()

if __name__ == "__main__":
    import sys
    main() 