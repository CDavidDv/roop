#!/usr/bin/env python3
"""
Script de instalación completo para ROOP en Google Colab
Instala el módulo roop y todas las dependencias necesarias
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_current_directory():
    """Verificar que estamos en el directorio correcto"""
    print("🔍 VERIFICANDO DIRECTORIO:")
    print("=" * 40)
    
    current_dir = os.getcwd()
    print(f"📁 Directorio actual: {current_dir}")
    
    # Verificar si estamos en el directorio de roop
    if os.path.exists('run.py') and os.path.exists('requirements.txt'):
        print("✅ Estamos en el directorio correcto de ROOP")
        return True
    else:
        print("❌ No estamos en el directorio de ROOP")
        print("📋 Asegúrate de ejecutar:")
        print("   !git clone https://github.com/s0md3v/roop.git")
        print("   %cd roop")
        return False

def install_roop_module():
    """Instalar el módulo roop en el entorno de Python"""
    print("\n📦 INSTALANDO MÓDULO ROOP:")
    print("=" * 40)
    
    try:
        # Verificar si el módulo roop existe
        if os.path.exists('roop'):
            print("✅ Directorio roop encontrado")
            
            # Instalar el módulo roop en modo desarrollo
            cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
            print(f"⏳ Ejecutando: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Módulo roop instalado exitosamente")
            return True
            
        else:
            print("❌ Directorio roop no encontrado")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando módulo roop:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def install_dependencies():
    """Instalar dependencias optimizadas para GPU"""
    print("\n📦 INSTALANDO DEPENDENCIAS:")
    print("=" * 40)
    
    try:
        # Instalar dependencias con índice extra de PyTorch
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118",
            "-r", "requirements.txt"
        ]
        
        print(f"⏳ Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencias instaladas exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def setup_python_path():
    """Configurar el path de Python para incluir el directorio actual"""
    print("\n🔧 CONFIGURANDO PYTHON PATH:")
    print("=" * 40)
    
    current_dir = os.getcwd()
    
    # Agregar el directorio actual al path de Python
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Agregado al path: {current_dir}")
    
    # Verificar que roop está disponible
    try:
        import roop
        print("✅ Módulo roop importado exitosamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando roop: {e}")
        return False

def test_roop_installation():
    """Probar que ROOP está instalado correctamente"""
    print("\n🧪 PROBANDO INSTALACIÓN ROOP:")
    print("=" * 40)
    
    try:
        # Probar importar roop
        import roop
        print("✅ Módulo roop importado")
        
        # Probar importar core
        from roop import core
        print("✅ Módulo core importado")
        
        # Probar importar otros módulos importantes
        try:
            from roop.processors.frame import face_swapper
            print("✅ Face swapper disponible")
        except ImportError:
            print("⚠️ Face swapper no disponible")
        
        try:
            from roop.processors.frame import face_enhancer
            print("✅ Face enhancer disponible")
        except ImportError:
            print("⚠️ Face enhancer no disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando ROOP: {e}")
        return False

def create_setup_py():
    """Crear archivo setup.py si no existe"""
    print("\n📝 CREANDO SETUP.PY:")
    print("=" * 40)
    
    setup_content = '''from setuptools import setup, find_packages

setup(
    name="roop",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorflow>=2.13.0",
        "onnxruntime-gpu>=1.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "insightface>=0.7.0",
        "gfpgan>=1.3.0",
        "basicsr>=1.4.0",
        "facexlib>=0.3.0",
        "filterpy>=1.4.0",
        "opennsfw2>=0.10.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "coloredlogs>=15.0.0",
        "humanfriendly>=10.0.0"
    ],
    python_requires=">=3.8",
)
'''
    
    try:
        with open('setup.py', 'w') as f:
            f.write(setup_content)
        print("✅ Archivo setup.py creado")
        return True
    except Exception as e:
        print(f"❌ Error creando setup.py: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("🚀 INSTALACIÓN COMPLETA DE ROOP PARA COLAB")
    print("=" * 60)
    
    # Verificar directorio
    if not check_current_directory():
        print("❌ Error: No estamos en el directorio correcto de ROOP")
        return False
    
    # Crear setup.py si no existe
    if not os.path.exists('setup.py'):
        create_setup_py()
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Instalar módulo roop
    if not install_roop_module():
        print("❌ Error instalando módulo roop")
        return False
    
    # Configurar path
    if not setup_python_path():
        print("❌ Error configurando path")
        return False
    
    # Probar instalación
    if not test_roop_installation():
        print("❌ Error probando instalación")
        return False
    
    print("\n✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Ejecutar: python optimize_colab_gpu.py")
    print("2. Usar: python run_colab_gpu_optimized.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Para lotes: python run_colab_gpu_optimized.py --source imagen.jpg --target carpeta_videos --batch --output-dir resultados")
    
    return True

if __name__ == '__main__':
    main() 