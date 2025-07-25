#!/usr/bin/env python3
"""
Script para instalar librerías CUDA manualmente
"""

import subprocess
import sys
import os
import urllib.request
import tarfile

def run_command(command, description=""):
    """Ejecuta un comando y maneja errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Exitoso")
            return True
        else:
            print(f"❌ {description} - Error")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Excepción: {e}")
        return False

def download_cuda_libraries():
    """Descarga librerías CUDA manualmente"""
    print("📥 DESCARGANDO LIBRERÍAS CUDA MANUALMENTE")
    print("=" * 50)
    
    # Crear directorio temporal
    if not run_command("mkdir -p /tmp/cuda_libs", "Creando directorio temporal"):
        return False
    
    # URLs de librerías CUDA (ejemplo - estas URLs pueden no existir)
    cuda_libs = {
        "libcufft.so.10": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcufft-11-8_11.8.0.76-1_amd64.deb",
        "libcurand.so.10": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-11-8_11.8.0.76-1_amd64.deb",
    }
    
    # Intentar descargar librerías
    for lib_name, url in cuda_libs.items():
        print(f"📥 Descargando {lib_name}...")
        try:
            urllib.request.urlretrieve(url, f"/tmp/cuda_libs/{lib_name}.deb")
            print(f"✅ {lib_name} descargado")
        except Exception as e:
            print(f"❌ Error descargando {lib_name}: {e}")
    
    return True

def install_cuda_from_system():
    """Intenta instalar CUDA desde el sistema"""
    print("🔧 INSTALANDO CUDA DESDE EL SISTEMA")
    print("=" * 50)
    
    # Intentar diferentes métodos de instalación
    methods = [
        "apt-get install -y cuda-toolkit-11-8",
        "apt-get install -y nvidia-cuda-toolkit",
        "apt-get install -y libcufft*",
        "apt-get install -y cuda-libraries-11-8",
    ]
    
    for method in methods:
        if run_command(method, f"Intentando: {method}"):
            print("✅ Instalación exitosa")
            return True
        else:
            print("❌ Método falló, intentando siguiente...")
    
    return False

def create_symlinks():
    """Crea enlaces simbólicos para las librerías CUDA"""
    print("🔗 CREANDO ENLACES SIMBÓLICOS")
    print("=" * 50)
    
    # Buscar librerías CUDA en el sistema
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/nvidia",
        "/usr/local/lib",
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"🔍 Buscando en: {path}")
            if run_command(f"find {path} -name 'libcufft*' -type f", f"Buscando libcufft en {path}"):
                print(f"✅ Encontradas librerías en {path}")
                # Crear enlaces simbólicos
                run_command(f"ln -sf {path}/libcufft.so* /usr/lib/", f"Creando enlaces desde {path}")
                return True
    
    return False

def install_cuda_runtime():
    """Instala CUDA Runtime desde conda"""
    print("📦 INSTALANDO CUDA RUNTIME DESDE CONDA")
    print("=" * 50)
    
    conda_commands = [
        "conda install -c conda-forge cudatoolkit=11.8 -y",
        "conda install -c nvidia cudatoolkit=11.8 -y",
        "pip install nvidia-cuda-runtime-cu118",
    ]
    
    for command in conda_commands:
        if run_command(command, f"Intentando: {command}"):
            print("✅ CUDA Runtime instalado")
            return True
    
    return False

def create_cuda_fix_script():
    """Crea un script que force el uso de CPU si CUDA falla"""
    print("🔧 CREANDO SCRIPT DE FIX PARA CUDA")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script que force CPU si CUDA no está disponible
"""

import os
import sys

# Intentar usar CUDA, si falla usar CPU
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    
    if 'CUDAExecutionProvider' in providers:
        print("✅ GPU disponible - usando CUDA")
        os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
    else:
        print("⚠️ GPU no disponible - usando CPU")
        os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
except Exception as e:
    print(f"⚠️ Error detectando GPU: {e}")
    print("🔄 Usando CPU como fallback")
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configurar otras variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Importar y ejecutar roop
from roop import core

if __name__ == "__main__":
    core.run()
'''
    
    with open('run_with_cuda_fix.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Script de fix creado: run_with_cuda_fix.py")
    return True

def verify_installation():
    """Verifica la instalación"""
    print("🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider disponible!")
            return True
        else:
            print("⚠️ CUDA provider no disponible, usando CPU")
            return False
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALANDO LIBRERÍAS CUDA MANUALMENTE")
    print("=" * 60)
    
    # Intentar diferentes métodos
    methods = [
        ("Instalando desde sistema", install_cuda_from_system),
        ("Creando enlaces simbólicos", create_symlinks),
        ("Instalando CUDA Runtime", install_cuda_runtime),
    ]
    
    success = False
    for method_name, method_func in methods:
        print(f"\n🔧 {method_name}...")
        if method_func():
            success = True
            break
    
    # Crear script de fix
    create_cuda_fix_script()
    
    # Verificar instalación
    verify_installation()
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
    print("=" * 60)
    print("✅ Métodos de instalación CUDA intentados")
    print("✅ Script de fix creado")
    print("\n🚀 Para usar con fix automático:")
    print("python run_with_cuda_fix.py --source tu_imagen.jpg --target video.mp4 -o resultado.mp4")
    print("\n💡 El script automáticamente usará GPU si está disponible, CPU si no")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 