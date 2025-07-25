#!/usr/bin/env python3
"""
Script para instalar libcudart y optimizar memoria para procesamiento
"""

import os
import sys
import subprocess
import shutil

def install_cudart():
    """Instala libcudart.so.11.0"""
    print("🔧 INSTALANDO LIBCUDART")
    print("=" * 50)
    
    commands = [
        "apt-get update",
        "apt-get install -y cuda-runtime-11-8",
        "apt-get install -y cuda-cudart-11-8",
        "ldconfig"
    ]
    
    for cmd in commands:
        print(f"🔄 Ejecutando: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ Completado: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error en {cmd}: {e.stderr}")
    
    # Crear enlace simbólico para libcudart
    try:
        cudart_source = "/usr/local/cuda-11.8/lib64/libcudart.so.11.0"
        cudart_target = "/usr/lib/x86_64-linux-gnu/libcudart.so.11.0"
        
        if os.path.exists(cudart_source):
            if os.path.exists(cudart_target):
                os.remove(cudart_target)
            print(f"🔗 Creando enlace: {cudart_source} -> {cudart_target}")
            subprocess.run(["ln", "-sf", cudart_source, cudart_target], check=True)
            print("✅ Enlace libcudart creado")
        else:
            print("⚠️ libcudart.so.11.0 no encontrada en CUDA")
    except Exception as e:
        print(f"❌ Error creando enlace libcudart: {e}")

def optimize_memory():
    """Optimiza la configuración de memoria"""
    print("⚡ OPTIMIZANDO MEMORIA")
    print("=" * 50)
    
    # Configurar variables de entorno para optimizar memoria
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
    
    # Configurar límites de memoria más conservadores
    os.environ['TF_MEMORY_ALLOCATION'] = '0.8'  # Usar solo 80% de memoria GPU
    os.environ['ONNXRUNTIME_GPU_MEMORY_LIMIT'] = '2147483648'  # 2GB límite
    
    print("✅ Configuración de memoria optimizada")

def test_cudart():
    """Prueba que libcudart esté disponible"""
    print("🧪 PROBANDO LIBCUDART")
    print("=" * 50)
    
    test_code = """
import ctypes
import os

# Configurar variables de entorno
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    # Intentar cargar libcudart
    ctypes.CDLL("libcudart.so.11.0")
    print("✅ libcudart.so.11.0 cargada correctamente")
    
    # Verificar librerías CUDA
    ctypes.CDLL("libcufft.so.10")
    print("✅ libcufft.so.10 cargada correctamente")
    
    ctypes.CDLL("libcublas.so.11")
    print("✅ libcublas.so.11 cargada correctamente")
    
    print("✅ Todas las librerías CUDA están disponibles")
    return True
except Exception as e:
    print(f"❌ Error cargando librerías: {e}")
    return False
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def run_processing_optimized():
    """Ejecuta el procesamiento con configuración optimizada"""
    print("🚀 EJECUTANDO PROCESAMIENTO OPTIMIZADO")
    print("=" * 60)
    
    # Configurar variables de entorno optimizadas
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
    os.environ['TF_MEMORY_ALLOCATION'] = '0.8'
    os.environ['ONNXRUNTIME_GPU_MEMORY_LIMIT'] = '2147483648'
    
    # Comando de procesamiento con configuración optimizada
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130_OPTIMIZED.mp4",
        "--frame-processor", "face_swapper",
        "--execution-provider", "cuda",
        "--execution-threads", "16",  # Menos hilos para evitar problemas de memoria
        "--temp-frame-quality", "85",  # Calidad ligeramente menor
        "--max-memory", "4",  # Menos memoria
        "--gpu-memory-wait", "60",  # Más tiempo de espera
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento optimizado...")
        print("⚠️ Configuración conservadora para evitar problemas de memoria")
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
    print("🚀 INSTALANDO CUDART Y OPTIMIZANDO PROCESAMIENTO")
    print("=" * 70)
    
    # Paso 1: Instalar libcudart
    install_cudart()
    
    # Paso 2: Optimizar memoria
    optimize_memory()
    
    # Paso 3: Probar librerías
    if not test_cudart():
        print("⚠️ Algunas librerías CUDA no están disponibles")
        print("🔄 Intentando procesamiento de todas formas...")
    
    # Paso 4: Ejecutar procesamiento optimizado
    success = run_processing_optimized()
    
    if success:
        print("🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
        print("📁 Archivo guardado en: /content/resultados/DanielaAS130_OPTIMIZED.mp4")
    else:
        print("❌ Error en el procesamiento")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 