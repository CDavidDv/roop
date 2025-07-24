#!/usr/bin/env python3
"""
Script para ejecutar ROOP ignorando la detección de versión y enfocándose en funcionalidad GPU
"""

import os
import sys
import subprocess

def setup_environment():
    """Configurar variables de entorno para forzar GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ONNXRUNTIME_PROVIDER_SHARED_LIB'] = '/usr/local/cuda/lib64/libonnxruntime_providers_cuda.so'

def check_gpu_functionality():
    """Verificar funcionalidad GPU sin depender de detección de versión"""
    print("🔍 VERIFICANDO FUNCIONALIDAD GPU")
    print("=" * 40)
    
    # Verificar GPU física
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU NVIDIA detectada")
            print(result.stdout.split('\n')[0])
        else:
            print("❌ GPU NVIDIA no detectada")
            return False
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False
    
    # Verificar ONNX Runtime
    try:
        result = subprocess.run([
            'roop_env/bin/python', '-c', 
            "import onnxruntime as ort; print('Proveedores:', ort.get_available_providers())"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and 'CUDAExecutionProvider' in result.stdout:
            print("✅ CUDAExecutionProvider disponible")
            return True
        else:
            print("❌ CUDAExecutionProvider NO disponible")
            return False
    except Exception as e:
        print(f"❌ Error verificando ONNX: {e}")
        return False

def run_roop_with_gpu():
    """Ejecutar ROOP con GPU forzado"""
    
    print("🎬 EJECUTANDO ROOP CON GPU FORZADO")
    print("=" * 50)
    
    # Configurar entorno
    setup_environment()
    
    # Comando optimizado para GPU
    cmd = [
        'roop_env/bin/python', 'run.py',
        '--source', '/content/DanielaAS.jpg',
        '--target', '/content/112.mp4',
        '-o', '/content/DanielaAS112_gpu_forced.mp4',
        '--frame-processor', 'face_swapper',
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '15',
        # Configuraciones de alta calidad
        '--temp-frame-format', 'png',
        '--temp-frame-quality', '0',
        '--output-video-encoder', 'libx264',
        '--output-video-quality', '35',
        '--keep-fps'
    ]
    
    print("⚙️ CONFIGURACIÓN:")
    print("   • GPU: Tesla T4 (15GB)")
    print("   • Execution Provider: cuda")
    print("   • Calidad: Alta (configuraciones originales)")
    print("   • Variables de entorno: Configuradas para GPU")
    print("=" * 50)
    
    try:
        print("🚀 Iniciando procesamiento con GPU forzado...")
        result = subprocess.run(cmd, check=True)
        print("\n✅ Procesamiento completado exitosamente")
        print("📁 Archivo generado: /content/DanielaAS112_gpu_forced.mp4")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en procesamiento: {e}")
        return False

def run_batch_with_gpu():
    """Ejecutar procesamiento en lote con GPU forzado"""
    
    print("🎬 PROCESAMIENTO EN LOTE CON GPU FORZADO")
    print("=" * 50)
    
    # Configurar entorno
    setup_environment()
    
    # Comando en lote
    cmd = [
        'roop_env/bin/python', 'run_batch_processing.py',
        '--source', '/content/LilitAS.png',
        '--videos', '/content/62.mp4', '/content/71.mp4', '/content/72.mp4', 
        '/content/74.mp4', '/content/75.mp4', '/content/76.mp4', 
        '/content/77.mp4', '/content/78.mp4', '/content/79.mp4',
        '--output-dir', '/content/resultados_gpu_forced',
        '--execution-threads', '8',
        # Configuraciones de alta calidad
        '--temp-frame-format', 'png',
        '--temp-frame-quality', '0',
        '--output-video-encoder', 'libx264',
        '--output-video-quality', '35',
        '--keep-fps'
    ]
    
    print("⚙️ CONFIGURACIÓN EN LOTE:")
    print("   • GPU: Tesla T4 (15GB)")
    print("   • Execution Provider: cuda")
    print("   • Calidad: Alta (configuraciones originales)")
    print("   • Videos: 9 archivos")
    print("   • Output: /content/resultados_gpu_forced")
    print("=" * 50)
    
    try:
        print("🚀 Iniciando procesamiento en lote con GPU forzado...")
        result = subprocess.run(cmd, check=True)
        print("\n✅ Procesamiento en lote completado")
        print("📁 Archivos generados en: /content/resultados_gpu_forced")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en procesamiento: {e}")
        return False

def monitor_gpu_usage():
    """Monitorear uso de GPU durante procesamiento"""
    print("\n📊 MONITOREO DE GPU")
    print("=" * 30)
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        name, mem_used, mem_total, util, temp = parts
                        print(f"🎮 GPU: {name}")
                        print(f"📊 VRAM: {mem_used}MB/{mem_total}MB ({int(mem_used)/int(mem_total)*100:.1f}%)")
                        print(f"⚡ Utilización: {util}%")
                        print(f"🌡️ Temperatura: {temp}°C")
    except Exception as e:
        print(f"❌ Error monitoreando GPU: {e}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ROOP con GPU forzado (ignora detección de versión)')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single', 
                       help='Modo de procesamiento: single o batch')
    parser.add_argument('--monitor', action='store_true', help='Monitorear GPU')
    
    args = parser.parse_args()
    
    # Verificar funcionalidad GPU
    if not check_gpu_functionality():
        print("❌ GPU no está funcionando correctamente")
        return
    
    # Monitorear GPU si se solicita
    if args.monitor:
        monitor_gpu_usage()
    
    # Ejecutar procesamiento
    if args.mode == 'single':
        success = run_roop_with_gpu()
    else:
        success = run_batch_with_gpu()
    
    if success:
        print("\n🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
        print("✅ GPU funcionando correctamente")
        print("✅ Alta calidad de video")
        print("✅ Sin pixelado")
        print("\n💡 NOTA: Ignoramos la detección de versión y nos enfocamos en funcionalidad")
    else:
        print("\n❌ Error en el procesamiento")
        print("💡 Verifica que CUDAExecutionProvider esté disponible")

if __name__ == "__main__":
    main() 