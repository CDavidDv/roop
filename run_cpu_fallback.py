#!/usr/bin/env python3
"""
Script para ejecutar procesamiento con CPU como fallback
"""

import os
import sys
import subprocess

def run_cpu_processing():
    """Ejecuta el procesamiento usando CPU"""
    print("🚀 EJECUTANDO PROCESAMIENTO CON CPU")
    print("=" * 50)
    
    # Configurar variables de entorno para CPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar GPU
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'
    
    # Comando de procesamiento con CPU
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130_CPU.mp4",
        "--frame-processor", "face_swapper",
        "--execution-provider", "cpu",
        "--execution-threads", "8",  # Menos hilos para CPU
        "--temp-frame-quality", "85",  # Calidad ligeramente menor
        "--max-memory", "4",  # Menos memoria
        "--gpu-memory-wait", "30",
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento con CPU...")
        print("⚠️ Nota: El procesamiento será más lento pero más estable")
        result = subprocess.run(command, timeout=7200)  # 2 horas timeout para CPU
        if result.returncode == 0:
            print("✅ Procesamiento completado exitosamente con CPU")
            return True
        else:
            print("❌ Error en procesamiento con CPU")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Timeout del procesamiento con CPU")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 PROCESAMIENTO CON CPU COMO FALLBACK")
    print("=" * 60)
    print("💡 Usando CPU porque CUDA no está disponible")
    print("⚡ El procesamiento será más lento pero más estable")
    print("=" * 60)
    
    success = run_cpu_processing()
    
    if success:
        print("🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
        print("📁 Archivo guardado en: /content/resultados/DanielaAS130_CPU.mp4")
    else:
        print("❌ Error en el procesamiento")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 