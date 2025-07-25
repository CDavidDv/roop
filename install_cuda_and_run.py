#!/usr/bin/env python3
"""
Script para instalar librerías CUDA y ejecutar procesamiento
"""

import os
import sys
import subprocess
import time

def install_cuda_libraries():
    """Instala las librerías CUDA necesarias"""
    print("🔧 INSTALANDO LIBRERÍAS CUDA")
    print("=" * 50)
    
    commands = [
        "apt-get update",
        "apt-get install -y libcufft-11-8 libcufft-dev-11-8",
        "apt-get install -y libcublas-11-8 libcublas-dev-11-8",
        "apt-get install -y libcudnn8 libcudnn8-dev",
        "ldconfig"
    ]
    
    for cmd in commands:
        print(f"🔄 Ejecutando: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ Completado: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error en {cmd}: {e.stderr}")
            # Continuar con el siguiente comando
    
    print("✅ Instalación de librerías CUDA completada")

def check_cuda_libraries():
    """Verifica que las librerías CUDA estén disponibles"""
    print("🔍 VERIFICANDO LIBRERÍAS CUDA")
    print("=" * 50)
    
    libraries = [
        "libcufft.so.10",
        "libcublas.so.11",
        "libcudnn.so.8"
    ]
    
    missing = []
    for lib in libraries:
        try:
            result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"✅ {lib} encontrada")
            else:
                print(f"❌ {lib} NO encontrada")
                missing.append(lib)
        except Exception as e:
            print(f"❌ Error verificando {lib}: {e}")
            missing.append(lib)
    
    if missing:
        print(f"⚠️ Librerías faltantes: {missing}")
        return False
    else:
        print("✅ Todas las librerías CUDA están disponibles")
        return True

def run_processing():
    """Ejecuta el procesamiento después de instalar las librerías"""
    print("🚀 EJECUTANDO PROCESAMIENTO CON GPU")
    print("=" * 50)
    
    # Configurar variables de entorno
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
    
    # Comando de procesamiento
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130.mp4",
        "--frame-processor", "face_swapper",
        "--execution-provider", "cuda",
        "--execution-threads", "24",
        "--temp-frame-quality", "90",
        "--max-memory", "8",
        "--gpu-memory-wait", "45",
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento...")
        result = subprocess.run(command, timeout=3600)  # 1 hora timeout
        if result.returncode == 0:
            print("✅ Procesamiento completado exitosamente")
            return True
        else:
            print("❌ Error en procesamiento")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Timeout del procesamiento")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALANDO CUDA Y EJECUTANDO PROCESAMIENTO")
    print("=" * 60)
    
    # Paso 1: Instalar librerías CUDA
    install_cuda_libraries()
    
    # Paso 2: Verificar librerías
    if not check_cuda_libraries():
        print("⚠️ Algunas librerías CUDA no están disponibles")
        print("🔄 Intentando continuar de todas formas...")
    
    # Paso 3: Ejecutar procesamiento
    print("🔄 Esperando 5 segundos para que las librerías se carguen...")
    time.sleep(5)
    
    success = run_processing()
    
    if success:
        print("🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
    else:
        print("❌ Error en el procesamiento")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 