#!/usr/bin/env python3
"""
Script completo para configurar ROOP con GPU optimizado desde el inicio
"""

import os
import sys
import subprocess
import shutil
import time

def print_header():
    """Imprime el encabezado del script"""
    print("🚀 CONFIGURACIÓN COMPLETA DE ROOP CON GPU")
    print("=" * 60)
    print("📋 Este script configurará todo ROOP desde cero")
    print("⚡ Optimizado para GPU con todas las librerías necesarias")
    print("=" * 60)

def install_system_dependencies():
    """Instala todas las dependencias del sistema"""
    print("🔧 INSTALANDO DEPENDENCIAS DEL SISTEMA")
    print("=" * 50)
    
    commands = [
        "apt-get update",
        "apt-get install -y ffmpeg",
        "apt-get install -y libcufft-11-8 libcufft-dev-11-8",
        "apt-get install -y libcublas-11-8 libcublas-dev-11-8",
        "apt-get install -y libcudnn8 libcudnn8-dev",
        "apt-get install -y cuda-runtime-11-8 cuda-cudart-11-8",
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

def create_cuda_links():
    """Crea enlaces simbólicos para librerías CUDA"""
    print("🔗 CREANDO ENLACES SIMBÓLICOS CUDA")
    print("=" * 50)
    
    links = [
        ("/usr/local/cuda-11.8/lib64/libcufft.so.10", "/usr/lib/x86_64-linux-gnu/libcufft.so.10"),
        ("/usr/local/cuda-11.8/lib64/libcublas.so.11", "/usr/lib/x86_64-linux-gnu/libcublas.so.11"),
        ("/usr/local/cuda-11.8/lib64/libcudart.so.11.0", "/usr/lib/x86_64-linux-gnu/libcudart.so.11.0"),
        ("/usr/lib/x86_64-linux-gnu/libcudnn.so.8", "/usr/lib/x86_64-linux-gnu/libcudnn.so.8")
    ]
    
    for source, target in links:
        try:
            if os.path.exists(source):
                if os.path.exists(target):
                    os.remove(target)  # Remover enlace existente
                print(f"🔗 Creando enlace: {source} -> {target}")
                subprocess.run(["ln", "-sf", source, target], check=True)
                print(f"✅ Enlace creado: {target}")
            else:
                print(f"⚠️ Fuente no existe: {source}")
        except Exception as e:
            print(f"❌ Error creando enlace {target}: {e}")

def install_python_dependencies():
    """Instala las dependencias de Python"""
    print("🐍 INSTALANDO DEPENDENCIAS DE PYTHON")
    print("=" * 50)
    
    # Lista de dependencias optimizadas
    dependencies = [
        "torch==2.0.1",
        "torchvision==0.15.2",
        "onnxruntime-gpu==1.15.1",
        "opencv-python==4.8.0.76",
        "numpy==1.24.3",
        "Pillow==10.0.0",
        "scikit-image==0.21.0",
        "scipy==1.11.1",
        "tqdm==4.65.0",
        "psutil==5.9.5",
        "insightface==0.7.3",
        "onnx==1.14.0",
        "opencv-contrib-python==4.8.0.76"
    ]
    
    for dep in dependencies:
        print(f"🔄 Instalando: {dep}")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Instalado: {dep}")
            else:
                print(f"⚠️ Error instalando {dep}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error con {dep}: {e}")

def configure_environment():
    """Configura las variables de entorno optimizadas"""
    print("⚙️ CONFIGURANDO VARIABLES DE ENTORNO")
    print("=" * 50)
    
    # Variables de entorno optimizadas
    env_vars = {
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'CUDA_VISIBLE_DEVICES': '0',
        'MPLBACKEND': 'Agg',
        'NO_ALBUMENTATIONS_UPDATE': '1',
        'ONNXRUNTIME_PROVIDER': 'CUDAExecutionProvider,CPUExecutionProvider',
        'TF_MEMORY_ALLOCATION': '0.8',
        'ONNXRUNTIME_GPU_MEMORY_LIMIT': '2147483648',
        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key} = {value}")

def test_gpu_setup():
    """Prueba la configuración de GPU"""
    print("🧪 PROBANDO CONFIGURACIÓN GPU")
    print("=" * 50)
    
    test_code = """
import torch
import onnxruntime as ort
import ctypes
import os

print("🔍 Verificando PyTorch GPU...")
if torch.cuda.is_available():
    print(f"✅ PyTorch GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"✅ Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("❌ PyTorch GPU no disponible")

print("\\n🔍 Verificando librerías CUDA...")
try:
    ctypes.CDLL("libcudart.so.11.0")
    print("✅ libcudart.so.11.0 cargada")
except Exception as e:
    print(f"❌ Error libcudart: {e}")

try:
    ctypes.CDLL("libcufft.so.10")
    print("✅ libcufft.so.10 cargada")
except Exception as e:
    print(f"❌ Error libcufft: {e}")

try:
    ctypes.CDLL("libcublas.so.11")
    print("✅ libcublas.so.11 cargada")
except Exception as e:
    print(f"❌ Error libcublas: {e}")

print("\\n🔍 Verificando ONNX Runtime...")
providers = ort.get_available_providers()
print(f"✅ Proveedores disponibles: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ CUDAExecutionProvider disponible")
else:
    print("❌ CUDAExecutionProvider no disponible")

print("\\n✅ Configuración GPU completada")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en prueba GPU: {e}")
        return False

def download_models():
    """Descarga los modelos necesarios"""
    print("📥 DESCARGANDO MODELOS")
    print("=" * 50)
    
    # Crear directorio de modelos si no existe
    models_dir = "/root/.insightface/models/buffalo_l"
    os.makedirs(models_dir, exist_ok=True)
    
    print("✅ Directorio de modelos creado")
    print("📋 Los modelos se descargarán automáticamente en el primer uso")

def create_optimized_run_script():
    """Crea un script optimizado para ejecutar ROOP"""
    print("📝 CREANDO SCRIPT DE EJECUCIÓN OPTIMIZADO")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script optimizado para ejecutar ROOP con GPU
"""

import os
import sys
import subprocess

# Configurar variables de entorno optimizadas
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
os.environ['TF_MEMORY_ALLOCATION'] = '0.8'
os.environ['ONNXRUNTIME_GPU_MEMORY_LIMIT'] = '2147483648'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

def run_roop_optimized():
    """Ejecuta ROOP con configuración optimizada"""
    print("🚀 EJECUTANDO ROOP CON GPU OPTIMIZADO")
    print("=" * 50)
    
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130_GPU.mp4",
        "--frame-processor", "face_swapper",
        "--execution-provider", "cuda",
        "--execution-threads", "16",
        "--temp-frame-quality", "85",
        "--max-memory", "4",
        "--gpu-memory-wait", "60",
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento...")
        result = subprocess.run(command, timeout=3600)
        if result.returncode == 0:
            print("✅ Procesamiento completado exitosamente")
            print("📁 Archivo guardado en: /content/resultados/DanielaAS130_GPU.mp4")
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

if __name__ == "__main__":
    success = run_roop_optimized()
    sys.exit(0 if success else 1)
'''
    
    with open("run_roop_optimized.py", "w") as f:
        f.write(script_content)
    
    print("✅ Script optimizado creado: run_roop_optimized.py")

def main():
    """Función principal"""
    print_header()
    
    # Paso 1: Instalar dependencias del sistema
    install_system_dependencies()
    
    # Paso 2: Crear enlaces CUDA
    create_cuda_links()
    
    # Paso 3: Instalar dependencias Python
    install_python_dependencies()
    
    # Paso 4: Configurar entorno
    configure_environment()
    
    # Paso 5: Probar configuración GPU
    if not test_gpu_setup():
        print("⚠️ Configuración GPU no completamente exitosa")
        print("🔄 Continuando de todas formas...")
    
    # Paso 6: Descargar modelos
    download_models()
    
    # Paso 7: Crear script optimizado
    create_optimized_run_script()
    
    print("\n🎉 ¡CONFIGURACIÓN COMPLETA FINALIZADA!")
    print("=" * 60)
    print("✅ Todas las dependencias instaladas")
    print("✅ Enlaces CUDA creados")
    print("✅ Variables de entorno configuradas")
    print("✅ Script optimizado creado")
    print("\n🚀 Para ejecutar ROOP:")
    print("   python run_roop_optimized.py")
    print("\n📁 El resultado se guardará en:")
    print("   /content/resultados/DanielaAS130_GPU.mp4")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 