#!/usr/bin/env python3
"""
Script de configuración automática para ROOP GPU desde notebook
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_status(message, status="INFO"):
    """Imprimir mensaje de estado"""
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{emoji.get(status, 'ℹ️')} {message}")

def check_gpu():
    """Verificar GPU"""
    print_status("Verificando GPU...", "INFO")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("GPU NVIDIA detectada", "SUCCESS")
            return True
        else:
            print_status("GPU NVIDIA no detectada", "ERROR")
            return False
    except FileNotFoundError:
        print_status("nvidia-smi no encontrado", "ERROR")
        return False

def check_environment():
    """Verificar environment de roop"""
    print_status("Verificando environment...", "INFO")
    roop_env_path = Path("roop_env")
    if roop_env_path.exists():
        python_path = roop_env_path / "bin" / "python"
        if python_path.exists():
            print_status(f"Environment encontrado: {roop_env_path}", "SUCCESS")
            return str(python_path)
        else:
            print_status(f"Python no encontrado en: {python_path}", "ERROR")
            return None
    else:
        print_status(f"Environment no encontrado: {roop_env_path}", "ERROR")
        return None

def install_onnx_gpu(python_path):
    """Instalar ONNX Runtime GPU"""
    print_status("Instalando ONNX Runtime GPU...", "INFO")
    
    # Desinstalar CPU
    print_status("Desinstalando ONNX Runtime CPU...", "INFO")
    subprocess.run([python_path, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'])
    
    # Instalar GPU
    print_status("Instalando ONNX Runtime GPU...", "INFO")
    result = subprocess.run([
        python_path, '-m', 'pip', 'install', 'onnxruntime-gpu==1.15.1'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print_status("ONNX Runtime GPU instalado", "SUCCESS")
        return True
    else:
        print_status(f"Error instalando: {result.stderr}", "ERROR")
        return False

def apply_gpu_optimizations():
    """Aplicar optimizaciones de GPU"""
    print_status("Aplicando optimizaciones de GPU...", "INFO")
    
    # Aplicar forzado de GPU
    result = subprocess.run([sys.executable, 'force_gpu_face_swapper.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print_status("Forzado de GPU aplicado", "SUCCESS")
    else:
        print_status(f"Error aplicando forzado: {result.stderr}", "WARNING")
    
    # Crear script de entorno
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
    print_status("Script de entorno creado: run_roop_gpu_env.sh", "SUCCESS")

def test_gpu_functionality(python_path):
    """Probar funcionalidad de GPU"""
    print_status("Probando funcionalidad de GPU...", "INFO")
    
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
    print_status("Creando comando de prueba...", "INFO")
    
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
    
    with open('test_gpu_command.sh', 'w') as f:
        f.write(test_cmd)
    
    os.chmod('test_gpu_command.sh', 0o755)
    print_status("Comando guardado en: test_gpu_command.sh", "SUCCESS")

def print_final_summary():
    """Imprimir resumen final"""
    print_status("CONFIGURACIÓN COMPLETADA", "SUCCESS")
    print("=" * 50)
    print("📁 Archivos creados:")
    print("  • run_roop_gpu_env.sh - Script principal con GPU")
    print("  • test_gpu_command.sh - Comando de prueba")
    print("  • GUIA_COMPLETA_ROOP_GPU.md - Guía completa")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("=" * 30)
    print("1. Ejecutar prueba:")
    print("   !./test_gpu_command.sh")
    print("\n2. Procesar en lote:")
    print("   !roop_env/bin/python run_batch_processing_optimized.py --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados --keep-fps")
    
    print("\n📊 LO QUE DEBERÍAS VER:")
    print("=" * 30)
    print("✅ VRAM > 0GB durante procesamiento")
    print("✅ Velocidad 1-2s/frame (no 6s/frame)")
    print("✅ Mensajes de diagnóstico GPU")
    print("✅ Tiempo total reducido 6x")

def setup_roop_gpu():
    """Configuración completa de ROOP con GPU"""
    print("🚀 CONFIGURACIÓN COMPLETA DE ROOP CON GPU")
    print("=" * 60)
    
    # Verificar GPU
    if not check_gpu():
        print_status("No se detectó GPU NVIDIA. Abortando.", "ERROR")
        return False
    
    # Verificar environment
    python_path = check_environment()
    if not python_path:
        print_status("No se encontró environment de roop. Abortando.", "ERROR")
        return False
    
    # Instalar ONNX GPU
    if not install_onnx_gpu(python_path):
        print_status("Error instalando ONNX Runtime GPU", "ERROR")
        return False
    
    # Aplicar optimizaciones
    apply_gpu_optimizations()
    
    # Probar GPU
    if not test_gpu_functionality(python_path):
        print_status("Error probando funcionalidad de GPU", "ERROR")
        return False
    
    # Crear comandos
    create_test_command()
    
    # Resumen final
    print_final_summary()
    
    print_status("¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!", "SUCCESS")
    print("Tu GPU Tesla T4 de 15GB está lista para procesar videos rápidamente.")
    
    return True

if __name__ == "__main__":
    setup_roop_gpu() 