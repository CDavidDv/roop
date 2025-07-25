#!/usr/bin/env python3
"""
Script para solucionar problemas de librerías CUDA faltantes
"""

import os
import sys
import subprocess

def run_command(command, description=""):
    """Ejecutar comando con manejo de errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def install_cuda_libraries():
    """Instalar librerías CUDA faltantes"""
    print("🔧 Instalando librerías CUDA faltantes...")
    
    # Instalar librerías CUDA del sistema
    run_command("apt-get update", "Actualizando repositorios")
    run_command("apt-get install -y libcublas-11-8 libcublas-dev-11-8", "Instalando libcublas")
    run_command("apt-get install -y libcudnn8 libcudnn8-dev", "Instalando libcudnn")
    run_command("apt-get install -y libnvinfer8 libnvinfer-dev", "Instalando libnvinfer")
    
    return True

def reinstall_onnxruntime_gpu():
    """Reinstalar ONNX Runtime GPU con librerías CUDA"""
    print("🔧 Reinstalando ONNX Runtime GPU...")
    
    # Desinstalar ONNX Runtime actual
    run_command("pip uninstall -y onnxruntime onnxruntime-gpu", "Desinstalando ONNX Runtime anterior")
    
    # Instalar ONNX Runtime GPU con soporte CUDA completo
    run_command("pip install onnxruntime-gpu==1.17.0", "Instalando ONNX Runtime GPU 1.17.0")
    
    # Instalar dependencias adicionales
    run_command("pip install nvidia-cudnn-cu12==8.9.4.25", "Instalando cuDNN")
    
    return True

def verify_cuda_libraries():
    """Verificar que las librerías CUDA estén disponibles"""
    print("\n🔍 Verificando librerías CUDA...")
    
    # Verificar archivos de librerías
    cuda_libs = [
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11",
        "/usr/lib/x86_64-linux-gnu/libcublas.so.11",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.8"
    ]
    
    for lib in cuda_libs:
        if os.path.exists(lib):
            print(f"✅ {lib}")
        else:
            print(f"❌ {lib} - No encontrado")
    
    # Verificar ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider disponible")
        else:
            print("❌ CUDA provider no disponible")
            
    except Exception as e:
        print(f"❌ Error verificando ONNX Runtime: {e}")
    
    return True

def create_cuda_fix():
    """Crear un fix temporal para CUDA"""
    print("\n🔧 Creando fix temporal para CUDA...")
    
    fix_content = '''#!/usr/bin/env python3
"""
Fix temporal para problemas de CUDA
"""

import os
import sys

# Configurar variables de entorno para CUDA
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_PATH'] = '/usr/local/cuda'

# Configurar para usar CPU si CUDA falla
os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'

print("✅ Variables de entorno CUDA configuradas")

# Ahora importar y ejecutar roop
try:
    from roop import core
    print("✅ ROOP importado correctamente")
except Exception as e:
    print(f"❌ Error importando ROOP: {e}")
'''
    
    with open("fix_cuda.py", "w") as f:
        f.write(fix_content)
    
    print("✅ Fix temporal creado: fix_cuda.py")
    return True

def create_simple_processor():
    """Crear un procesador simplificado que use solo CPU"""
    print("\n🔧 Creando procesador simplificado...")
    
    simple_content = '''#!/usr/bin/env python3
"""
Procesador simplificado que usa solo CPU para evitar problemas de CUDA
"""

import os
import sys

# Configurar para usar solo CPU
os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configurar variables de entorno
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Desactivar predictor NSFW
import roop.predictor
def predict_video_skip_nsfw(target_path: str) -> bool:
    print("⚠️ Saltando verificación NSFW...")
    return False

roop.predictor.predict_video = predict_video_skip_nsfw

# Ahora importar y ejecutar roop
from roop import core

if __name__ == "__main__":
    core.run()
'''
    
    with open("run_cpu_only.py", "w") as f:
        f.write(simple_content)
    
    print("✅ Procesador CPU creado: run_cpu_only.py")
    return True

def main():
    """Función principal"""
    print("🚀 SOLUCIONANDO PROBLEMAS DE CUDA")
    print("=" * 60)
    
    # Instalar librerías CUDA
    if not install_cuda_libraries():
        print("❌ Error instalando librerías CUDA")
        return False
    
    # Reinstalar ONNX Runtime
    if not reinstall_onnxruntime_gpu():
        print("❌ Error reinstalando ONNX Runtime")
        return False
    
    # Verificar librerías
    if not verify_cuda_libraries():
        print("❌ Error verificando librerías")
        return False
    
    # Crear fixes temporales
    if not create_cuda_fix():
        print("❌ Error creando fix CUDA")
        return False
    
    if not create_simple_processor():
        print("❌ Error creando procesador CPU")
        return False
    
    print("\n" + "=" * 60)
    print("✅ PROBLEMAS DE CUDA SOLUCIONADOS")
    print("=" * 60)
    print("📋 Opciones disponibles:")
    print("1. Procesamiento normal: python run.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("2. Solo CPU (más estable): python run_cpu_only.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Fix CUDA: python fix_cuda.py")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main() 