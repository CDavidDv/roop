#!/usr/bin/env python3
"""
Script para arreglar dependencias de CUDA
"""

import subprocess
import sys
import os

def install_cuda_dependencies():
    """Instala las dependencias de CUDA faltantes"""
    print("🔧 ARREGLANDO DEPENDENCIAS DE CUDA")
    print("=" * 50)
    
    commands = [
        # Instalar librerías CUDA faltantes
        "apt-get update",
        "apt-get install -y libcufft10 libcudnn8 libcublas11",
        
        # Crear enlaces simbólicos si es necesario
        "ln -sf /usr/lib/x86_64-linux-gnu/libcufft.so.10 /usr/local/cuda-11.8/lib64/libcufft.so.10",
        "ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda-11.8/lib64/libcudnn.so.8",
        "ln -sf /usr/lib/x86_64-linux-gnu/libcublas.so.11 /usr/local/cuda-11.8/lib64/libcublas.so.11",
        
        # Actualizar variables de entorno
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH",
        
        # Verificar instalación
        "ldconfig",
    ]
    
    for cmd in commands:
        print(f"Ejecutando: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd}")
            else:
                print(f"⚠️ {cmd} - {result.stderr}")
        except Exception as e:
            print(f"❌ Error en {cmd}: {e}")

def check_cuda_libraries():
    """Verifica que las librerías CUDA estén disponibles"""
    print("\n🔍 VERIFICANDO LIBRERÍAS CUDA")
    print("=" * 40)
    
    libraries = [
        "/usr/lib/x86_64-linux-gnu/libcufft.so.10",
        "/usr/local/cuda-11.8/lib64/libcufft.so.10",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
        "/usr/local/cuda-11.8/lib64/libcudnn.so.8",
    ]
    
    for lib in libraries:
        if os.path.exists(lib):
            size = os.path.getsize(lib)
            print(f"✅ {lib} - {size:,} bytes")
        else:
            print(f"❌ {lib} - NO ENCONTRADO")
    
    # Verificar variables de entorno
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"\n📁 LD_LIBRARY_PATH: {ld_path}")

def fix_onnxruntime_cuda():
    """Arregla la configuración de ONNX Runtime para CUDA"""
    print("\n🔧 ARREGLANDO ONNX RUNTIME CUDA")
    print("=" * 40)
    
    # Reinstalar onnxruntime-gpu
    commands = [
        "pip uninstall -y onnxruntime onnxruntime-gpu",
        "pip install onnxruntime-gpu==1.16.3",
        "pip install --upgrade onnxruntime-gpu",
    ]
    
    for cmd in commands:
        print(f"Ejecutando: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd}")
            else:
                print(f"⚠️ {cmd} - {result.stderr}")
        except Exception as e:
            print(f"❌ Error en {cmd}: {e}")

def main():
    """Función principal"""
    print("🚀 ARREGLANDO DEPENDENCIAS DE CUDA")
    print("=" * 60)
    
    # Verificar si estamos en root
    if os.geteuid() != 0:
        print("⚠️ Este script necesita permisos de root para instalar librerías")
        print("Ejecuta con: sudo python fix_cuda_dependencies.py")
        return 1
    
    install_cuda_dependencies()
    check_cuda_libraries()
    fix_onnxruntime_cuda()
    
    print("\n🎉 ¡DEPENDENCIAS DE CUDA ARREGLADAS!")
    print("=" * 50)
    print("✅ Librerías CUDA instaladas")
    print("✅ ONNX Runtime GPU actualizado")
    print("✅ Variables de entorno configuradas")
    print("\n🚀 Ahora puedes ejecutar el procesamiento:")
    print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 