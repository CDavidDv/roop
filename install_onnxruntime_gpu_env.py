#!/usr/bin/env python3
"""
Script para instalar onnxruntime-gpu en el environment de roop
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_roop_environment():
    """Verificar si estamos en el environment de roop"""
    print("🔍 VERIFICANDO ENVIRONMENT DE ROOP")
    print("=" * 50)
    
    # Verificar si existe el environment
    roop_env_path = Path("roop_env")
    if roop_env_path.exists():
        print(f"✅ Environment encontrado: {roop_env_path}")
        
        # Verificar Python del environment
        python_path = roop_env_path / "bin" / "python"
        if python_path.exists():
            print(f"✅ Python del environment: {python_path}")
            return str(python_path)
        else:
            print(f"❌ Python no encontrado en: {python_path}")
            return None
    else:
        print(f"❌ Environment no encontrado: {roop_env_path}")
        return None

def check_current_onnx_in_env(python_path):
    """Verificar instalación actual de ONNX Runtime en el environment"""
    print("\n🔍 VERIFICANDO ONNX RUNTIME EN ENVIRONMENT")
    print("=" * 50)
    
    try:
        # Usar Python del environment para verificar
        result = subprocess.run([
            python_path, '-c', 
            "import onnxruntime as ort; print('Versión:', ort.__version__); print('GPU:', 'gpu' in ort.__version__.lower())"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            if 'True' in result.stdout:
                print("✅ ONNX Runtime GPU detectado en environment")
                return True
            else:
                print("❌ ONNX Runtime CPU detectado en environment")
                return False
        else:
            print(f"❌ Error verificando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def uninstall_onnxruntime_cpu_in_env(python_path):
    """Desinstalar ONNX Runtime CPU en el environment"""
    print("\n🗑️ DESINSTALANDO ONNX RUNTIME CPU EN ENVIRONMENT")
    print("=" * 50)
    
    try:
        # Desinstalar usando pip del environment
        result = subprocess.run([
            python_path, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ONNX Runtime CPU desinstalado del environment")
            return True
        else:
            print(f"⚠️ Error desinstalando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def install_onnxruntime_gpu_in_env(python_path):
    """Instalar ONNX Runtime GPU en el environment"""
    print("\n📦 INSTALANDO ONNX RUNTIME GPU EN ENVIRONMENT")
    print("=" * 50)
    
    try:
        # Instalar usando pip del environment
        result = subprocess.run([
            python_path, '-m', 'pip', 'install', 'onnxruntime-gpu==1.15.1'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ONNX Runtime GPU instalado en environment")
            return True
        else:
            print(f"❌ Error instalando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_gpu_in_env(python_path):
    """Verificar GPU en el environment"""
    print("\n✅ VERIFICANDO GPU EN ENVIRONMENT")
    print("=" * 40)
    
    try:
        # Verificar con Python del environment
        verify_script = '''
import onnxruntime as ort
import numpy as np

print(f"📦 Versión: {ort.__version__}")
providers = ort.get_available_providers()
print(f"🔧 Proveedores: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ CUDA GPU disponible en environment")
    
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
        
        print("✅ Sesión CUDA creada exitosamente en environment")
        print("✅ GPU funciona correctamente en environment")
        
    except Exception as e:
        print(f"⚠️ Error probando sesión CUDA: {e}")
        
else:
    print("❌ CUDA GPU NO disponible en environment")
'''
        
        result = subprocess.run([
            python_path, '-c', verify_script
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if '✅ CUDA GPU disponible en environment' in result.stdout:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error verificando: {e}")
        return False

def create_environment_script():
    """Crear script con variables de entorno para GPU en environment"""
    
    env_script = '''#!/bin/bash
# Variables de entorno para forzar GPU en ROOP (Environment)
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export ONNXRUNTIME_PROVIDER_SHARED_LIB=/usr/local/cuda/lib64/libonnxruntime_providers_cuda.so

echo "🔧 Variables de entorno configuradas para GPU (Environment)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Ejecutar ROOP con GPU usando Python del environment
roop_env/bin/python run.py "$@"
'''
    
    with open('run_roop_gpu_env.sh', 'w') as f:
        f.write(env_script)
    
    # Hacer ejecutable
    os.chmod('run_roop_gpu_env.sh', 0o755)
    
    print("\n📝 Script creado: run_roop_gpu_env.sh")
    print("💡 Uso: ./run_roop_gpu_env.sh --source imagen.jpg --target video.mp4 -o salida.mp4")

def test_face_swapper_gpu_env(python_path):
    """Probar face-swapper con GPU en environment"""
    print("\n🧪 PROBANDO FACE-SWAPPER CON GPU EN ENVIRONMENT")
    print("=" * 50)
    
    # Comando de prueba usando Python del environment
    test_cmd = [
        str(python_path), 'run.py',
        '--source', '/content/DanielaAS.jpg',
        '--target', '/content/112.mp4',
        '-o', '/content/test_gpu_face_swapper_env.mp4',
        '--frame-processor', 'face_swapper',
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '5',
        '--temp-frame-quality', '100',
        '--temp-frame-format', 'png',
        '--output-video-encoder', 'h264_nvenc',
        '--output-video-quality', '100',
        '--keep-fps'
    ]
    
    print("Comando de prueba (Environment):")
    print(" ".join(test_cmd))
    
    print("\n💡 Para probar:")
    print("1. Ejecuta el comando anterior")
    print("2. Monitorea con: nvidia-smi -l 1")
    print("3. Verifica que VRAM > 0GB durante el procesamiento")
    print("4. Busca mensajes de diagnóstico GPU en la salida")

def main():
    print("🔧 INSTALADOR DE ONNX RUNTIME GPU EN ENVIRONMENT")
    print("=" * 60)
    
    # Verificar environment
    python_path = check_roop_environment()
    if not python_path:
        print("❌ No se pudo encontrar el environment de roop")
        return
    
    # Verificar instalación actual en environment
    current_is_gpu = check_current_onnx_in_env(python_path)
    
    if current_is_gpu:
        print("\n✅ Ya tienes ONNX Runtime GPU instalado en el environment")
        verify_gpu_in_env(python_path)
        test_face_swapper_gpu_env(python_path)
        create_environment_script()
        return
    
    print("\n❌ Necesitas instalar ONNX Runtime GPU en el environment")
    print("=" * 50)
    
    # Confirmar instalación
    response = input("¿Instalar ONNX Runtime GPU en el environment? (y/n): ").lower()
    if response != 'y':
        print("❌ Instalación cancelada")
        return
    
    # Desinstalar CPU en environment
    if not uninstall_onnxruntime_cpu_in_env(python_path):
        print("⚠️ No se pudo desinstalar ONNX Runtime CPU del environment")
    
    # Instalar GPU en environment
    if install_onnxruntime_gpu_in_env(python_path):
        print("\n✅ Instalación completada en environment")
        
        # Verificar instalación en environment
        if verify_gpu_in_env(python_path):
            print("\n🎉 ¡ONNX Runtime GPU instalado correctamente en environment!")
            
            # Probar face-swapper en environment
            test_face_swapper_gpu_env(python_path)
            
            # Crear script de entorno
            create_environment_script()
            
            print("\n" + "=" * 60)
            print("🚀 PRÓXIMOS PASOS:")
            print("=" * 60)
            print("1. Usa el script: ./run_roop_gpu_env.sh")
            print("2. O ejecuta directamente:")
            print(f"   {python_path} run.py --source imagen.jpg --target video.mp4 -o salida.mp4")
            print("3. Monitorea con: nvidia-smi -l 1")
            print("4. Verifica que VRAM > 0GB durante el procesamiento")
        else:
            print("\n❌ Error verificando instalación de GPU en environment")
    else:
        print("\n❌ Error instalando ONNX Runtime GPU en environment")

if __name__ == "__main__":
    main() 