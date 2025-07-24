#!/usr/bin/env python3
"""
Script para verificar el uso real de GPU durante el procesamiento de ROOP
"""

import subprocess
import time
import psutil
from datetime import datetime

def check_gpu_usage():
    """Verificar uso real de GPU"""
    print("🔍 VERIFICACIÓN DE USO DE GPU")
    print("=" * 50)
    
    try:
        # Verificar nvidia-smi
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        print(f"🎮 GPU: {parts[0]}")
                        print(f"📊 VRAM usada: {parts[1]}MB / {parts[2]}MB")
                        print(f"📊 Utilización: {parts[3]}%")
                        print(f"🌡️ Temperatura: {parts[4]}°C")
                        
                        # Calcular porcentaje
                        used_mb = int(parts[1])
                        total_mb = int(parts[2])
                        percent = (used_mb / total_mb) * 100
                        print(f"📊 Porcentaje VRAM: {percent:.1f}%")
        else:
            print("❌ Error ejecutando nvidia-smi")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def check_processes_using_gpu():
    """Verificar procesos que están usando GPU"""
    print("\n🔍 PROCESOS USANDO GPU:")
    print("=" * 30)
    
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip():
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            print(f"PID {parts[0]}: {parts[1]} - {parts[2]}MB")
            else:
                print("❌ No hay procesos usando GPU")
        else:
            print("❌ Error ejecutando nvidia-smi")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def monitor_gpu_continuously():
    """Monitorear GPU continuamente"""
    print("\n📊 MONITOREO CONTINUO DE GPU")
    print("=" * 40)
    print("⏰ Actualizaciones cada 5 segundos...")
    print("Presiona Ctrl+C para detener")
    print("=" * 40)
    
    try:
        while True:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n⏰ {timestamp}")
            
            # Verificar GPU
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            used_mb = int(parts[0])
                            total_mb = int(parts[1])
                            utilization = int(parts[2])
                            percent = (used_mb / total_mb) * 100
                            
                            print(f"🎮 VRAM: {used_mb}MB / {total_mb}MB ({percent:.1f}%)")
                            print(f"📊 Utilización: {utilization}%")
                            
                            # Alertas
                            if percent > 90:
                                print("⚠️ ALERTA: VRAM muy alta!")
                            elif percent > 80:
                                print("⚠️ VRAM alta")
                            elif percent > 50:
                                print("✅ VRAM en uso activo")
                            else:
                                print("💤 VRAM en reposo")
            
            # Verificar procesos
            process_result = subprocess.run([
                'nvidia-smi', 
                '--query-compute-apps=process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if process_result.returncode == 0:
                lines = process_result.stdout.strip().split('\n')
                if lines and lines[0].strip():
                    print("🔍 Procesos GPU:")
                    total_memory = 0
                    for line in lines:
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 2:
                                process_name = parts[0]
                                memory_mb = int(parts[1])
                                total_memory += memory_mb
                                print(f"  • {process_name}: {memory_mb}MB")
                    print(f"📊 Total procesos GPU: {total_memory}MB")
                else:
                    print("🔍 No hay procesos usando GPU")
            
            print("-" * 40)
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoreo detenido")

def main():
    print("🔍 VERIFICADOR DE USO DE GPU PARA ROOP")
    print("=" * 50)
    
    # Verificación inicial
    check_gpu_usage()
    check_processes_using_gpu()
    
    print("\n" + "=" * 50)
    print("💡 CONSEJOS:")
    print("  • Si VRAM = 0GB: El procesador no está usando GPU")
    print("  • Si VRAM > 0GB: GPU está siendo utilizada")
    print("  • Monitorea durante el procesamiento para ver cambios")
    print("=" * 50)
    
    # Preguntar si quiere monitoreo continuo
    response = input("\n¿Iniciar monitoreo continuo? (y/n): ").lower()
    if response == 'y':
        monitor_gpu_continuously()
    else:
        print("✅ Verificación completada")

if __name__ == "__main__":
    main() 