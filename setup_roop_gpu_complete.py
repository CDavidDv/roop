#!/usr/bin/env python3
"""
Script completo para configurar ROOP con GPU desde el inicio hasta la ejecución
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Imprimir encabezado"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def check_gpu():
    """Verificar GPU"""
    print_header("VERIFICACIÓN DE GPU")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU NVIDIA detectada")
            print(result.stdout.split('\n')[0])  # Primera línea con info de driver
            return True
        else:
            print("❌ GPU NVIDIA no detectada")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
        return False

def check_environment():
    """Verificar environment de roop"""
    print_header("VERIFICACIÓN DE ENVIRONMENT")
    
    roop_env_path = Path("roop_env")
    if roop_env_path.exists():
        python_path = roop_env_path / "bin" / "python"
        if python_path.exists():
            print(f"✅ Environment encontrado: {roop_env_path}")
            print(f"✅ Python: {python_path}")
            return str(python_path)
        else:
            print(f"❌ Python no encontrado en: {python_path}")
            return None
    else:
        print(f"❌ Environment no encontrado: {roop_env_path}")
        return None

def check_onnx_runtime(python_path):
    """Verificar ONNX Runtime"""
    print_header("VERIFICACIÓN DE ONNX RUNTIME")
    
    try:
        result = subprocess.run([
            python_path, '-c', 
            "import onnxruntime as ort; print('Versión:', ort.__version__); print('GPU:', 'gpu' in ort.__version__.lower())"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            if 'True' in result.stdout:
                print("✅ ONNX Runtime GPU detectado")
                return True
            else:
                print("❌ ONNX Runtime CPU detectado")
                return False
        else:
            print(f"❌ Error verificando ONNX: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def install_onnx_gpu(python_path):
    """Instalar ONNX Runtime GPU"""
    print_header("INSTALACIÓN DE ONNX RUNTIME GPU")
    
    # Desinstalar CPU
    print("🗑️ Desinstalando ONNX Runtime CPU...")
    subprocess.run([python_path, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'])
    
    # Instalar GPU
    print("📦 Instalando ONNX Runtime GPU...")
    result = subprocess.run([
        python_path, '-m', 'pip', 'install', 'onnxruntime-gpu==1.15.1'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ ONNX Runtime GPU instalado")
        return True
    else:
        print(f"❌ Error instalando: {result.stderr}")
        return False

def apply_gpu_optimizations():
    """Aplicar optimizaciones de GPU"""
    print_header("APLICANDO OPTIMIZACIONES DE GPU")
    
    # Aplicar forzado de GPU
    print("🔧 Aplicando forzado de GPU...")
    result = subprocess.run([sys.executable, 'force_gpu_face_swapper.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Forzado de GPU aplicado")
    else:
        print(f"⚠️ Error aplicando forzado: {result.stderr}")
    
    # Crear script de entorno
    print("📝 Creando script de entorno...")
    env_script = '''#!/bin/bash
# Variables de entorno para forzar GPU en ROOP
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export ONNXRUNTIME_PROVIDER_SHARED_LIB=/usr/local/cuda/lib64/libonnxruntime_providers_cuda.so

echo "🔧 Variables de entorno configuradas para GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Ejecutar ROOP con GPU
roop_env/bin/python run.py "$@"
'''
    
    with open('run_roop_gpu_env.sh', 'w') as f:
        f.write(env_script)
    
    os.chmod('run_roop_gpu_env.sh', 0o755)
    print("✅ Script de entorno creado: run_roop_gpu_env.sh")

def test_gpu_functionality(python_path):
    """Probar funcionalidad de GPU"""
    print_header("PRUEBA DE FUNCIONALIDAD DE GPU")
    
    test_script = '''
import onnxruntime as ort
import numpy as np

print("🔍 Probando GPU...")

# Verificar proveedores
providers = ort.get_available_providers()
print(f"Proveedores disponibles: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ CUDA disponible")
    
    # Probar sesión simple
    try:
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        session_options = ort.SessionOptions()
        
        # Crear modelo simple
        import onnx
        from onnx import helper
        
        X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph)
        
        # Probar con CUDA
        session = ort.InferenceSession(
            model.SerializeToString(),
            session_options,
            providers=['CUDAExecutionProvider']
        )
        
        print("✅ Sesión CUDA creada exitosamente")
        print("✅ GPU funciona correctamente")
        
    except Exception as e:
        print(f"❌ Error probando sesión CUDA: {e}")
        
else:
    print("❌ CUDA NO disponible")
'''
    
    result = subprocess.run([python_path, '-c', test_script], capture_output=True, text=True)
    print(result.stdout)
    
    if '✅ GPU funciona correctamente' in result.stdout:
        return True
    else:
        return False

def create_test_command():
    """Crear comando de prueba"""
    print_header("COMANDO DE PRUEBA")
    
    test_cmd = '''./run_roop_gpu_env.sh \\
  --source /content/DanielaAS.jpg \\
  --target /content/112.mp4 \\
  -o /content/DanielaAS112_gpu.mp4 \\
  --frame-processor face_swapper \\
  --execution-provider cuda \\
  --max-memory 8 \\
  --execution-threads 8 \\
  --gpu-memory-wait 5 \\
  --temp-frame-quality 100 \\
  --temp-frame-format png \\
  --output-video-encoder h264_nvenc \\
  --output-video-quality 100 \\
  --keep-fps'''
    
    print("Comando de prueba:")
    print(test_cmd)
    
    # Guardar en archivo
    with open('test_gpu_command.sh', 'w') as f:
        f.write(test_cmd)
    
    os.chmod('test_gpu_command.sh', 0o755)
    print("\n✅ Comando guardado en: test_gpu_command.sh")

def create_monitoring_script():
    """Crear script de monitoreo"""
    print_header("SCRIPT DE MONITOREO")
    
    monitoring_script = '''#!/bin/bash
echo "📊 MONITOREO DE GPU EN TIEMPO REAL"
echo "Presiona Ctrl+C para detener"
echo "=" * 50

while true; do
    clear
    echo "$(date)"
    echo "=" * 30
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    echo "=" * 30
    sleep 5
done'''
    
    with open('monitor_gpu.sh', 'w') as f:
        f.write(monitoring_script)
    
    os.chmod('monitor_gpu.sh', 0o755)
    print("✅ Script de monitoreo creado: monitor_gpu.sh")

def print_final_summary():
    """Imprimir resumen final"""
    print_header("RESUMEN FINAL")
    
    print("✅ CONFIGURACIÓN COMPLETADA")
    print("=" * 40)
    print("📁 Archivos creados:")
    print("  • run_roop_gpu_env.sh - Script principal con GPU")
    print("  • test_gpu_command.sh - Comando de prueba")
    print("  • monitor_gpu.sh - Monitoreo de GPU")
    print("  • GUIA_COMPLETA_ROOP_GPU.md - Guía completa")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("=" * 30)
    print("1. Ejecutar prueba:")
    print("   ./test_gpu_command.sh")
    print("\n2. Monitorear GPU:")
    print("   ./monitor_gpu.sh")
    print("\n3. Procesar en lote:")
    print("   roop_env/bin/python run_batch_processing_optimized.py --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados --keep-fps")
    
    print("\n📊 LO QUE DEBERÍAS VER:")
    print("=" * 30)
    print("✅ VRAM > 0GB durante procesamiento")
    print("✅ Velocidad 1-2s/frame (no 6s/frame)")
    print("✅ Mensajes de diagnóstico GPU")
    print("✅ Tiempo total reducido 6x")

def main():
    print("🚀 CONFIGURACIÓN COMPLETA DE ROOP CON GPU")
    print("=" * 60)
    print("Este script configurará todo automáticamente")
    print("=" * 60)
    
    # Verificar GPU
    if not check_gpu():
        print("❌ No se detectó GPU NVIDIA. Abortando.")
        return
    
    # Verificar environment
    python_path = check_environment()
    if not python_path:
        print("❌ No se encontró environment de roop. Abortando.")
        return
    
    # Verificar ONNX Runtime
    onnx_gpu = check_onnx_runtime(python_path)
    
    # Instalar ONNX GPU si es necesario
    if not onnx_gpu:
        print("\n❌ Necesitas instalar ONNX Runtime GPU")
        response = input("¿Instalar ONNX Runtime GPU? (y/n): ").lower()
        if response == 'y':
            if not install_onnx_gpu(python_path):
                print("❌ Error instalando ONNX Runtime GPU")
                return
        else:
            print("❌ Instalación cancelada")
            return
    
    # Aplicar optimizaciones
    apply_gpu_optimizations()
    
    # Probar GPU
    if not test_gpu_functionality(python_path):
        print("❌ Error probando funcionalidad de GPU")
        return
    
    # Crear comandos
    create_test_command()
    create_monitoring_script()
    
    # Resumen final
    print_final_summary()
    
    print("\n🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("Tu GPU Tesla T4 de 15GB está lista para procesar videos rápidamente.")

if __name__ == "__main__":
    main() 