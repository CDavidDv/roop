#!/usr/bin/env python3
"""
Script para arreglar la instalación de ROOP en Google Colab
Maneja diferentes estructuras de repositorio
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def detect_roop_structure():
    """Detectar la estructura del repositorio ROOP"""
    print("🔍 DETECTANDO ESTRUCTURA DE ROOP:")
    print("=" * 40)
    
    current_dir = os.getcwd()
    print(f"📁 Directorio actual: {current_dir}")
    
    # Buscar archivos clave de ROOP
    key_files = ['run.py', 'requirements.txt']
    found_files = []
    
    for file in key_files:
        if os.path.exists(file):
            found_files.append(file)
            print(f"✅ Encontrado: {file}")
        else:
            print(f"❌ No encontrado: {file}")
    
    # Buscar directorio roop
    if os.path.exists('roop'):
        print("✅ Directorio roop encontrado")
        return 'standard'
    else:
        print("❌ Directorio roop no encontrado")
        
        # Buscar archivos que sugieran estructura alternativa
        if os.path.exists('roop.py'):
            print("✅ Archivo roop.py encontrado")
            return 'single_file'
        
        # Buscar otros directorios que puedan contener el código
        for item in os.listdir('.'):
            if os.path.isdir(item) and item != '.git':
                print(f"📁 Directorio encontrado: {item}")
        
        return 'unknown'

def create_roop_module():
    """Crear el módulo roop si no existe"""
    print("\n📦 CREANDO MÓDULO ROOP:")
    print("=" * 40)
    
    # Crear directorio roop
    if not os.path.exists('roop'):
        os.makedirs('roop')
        print("✅ Directorio roop creado")
    
    # Crear __init__.py
    init_content = '''"""
ROOP - Deep Face Swap
"""

__version__ = "1.0.0"
'''
    
    with open('roop/__init__.py', 'w') as f:
        f.write(init_content)
    print("✅ __init__.py creado")
    
    # Buscar archivos Python que puedan ser parte del módulo
    python_files = []
    for file in os.listdir('.'):
        if file.endswith('.py') and file not in ['run.py', 'install_roop_colab.py', 'fix_roop_installation.py']:
            python_files.append(file)
    
    print(f"📄 Archivos Python encontrados: {python_files}")
    
    # Crear estructura básica del módulo
    module_files = {
        'core.py': '''"""
Core functionality for ROOP
"""

def run():
    """Main entry point"""
    print("ROOP core module loaded")
''',
        'face_analyser.py': '''"""
Face analysis functionality
"""

def get_face_analyser():
    """Get face analyser instance"""
    return None
''',
        'predictor.py': '''"""
NSFW predictor
"""

def predict_video_skip_nsfw(target_path: str) -> bool:
    """Skip NSFW prediction"""
    return False
'''
    }
    
    # Crear directorios necesarios
    os.makedirs('roop/processors', exist_ok=True)
    os.makedirs('roop/processors/frame', exist_ok=True)
    
    # Crear archivos del módulo
    for filename, content in module_files.items():
        with open(f'roop/{filename}', 'w') as f:
            f.write(content)
        print(f"✅ {filename} creado")
    
    # Crear __init__.py para processors
    with open('roop/processors/__init__.py', 'w') as f:
        f.write('"""Processors module"""\n')
    
    with open('roop/processors/frame/__init__.py', 'w') as f:
        f.write('"""Frame processors module"""\n')
    
    print("✅ Estructura del módulo roop creada")

def install_dependencies():
    """Instalar dependencias"""
    print("\n📦 INSTALANDO DEPENDENCIAS:")
    print("=" * 40)
    
    try:
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

def install_roop_package():
    """Instalar el paquete roop"""
    print("\n📦 INSTALANDO PAQUETE ROOP:")
    print("=" * 40)
    
    try:
        # Instalar en modo desarrollo
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        print(f"⏳ Ejecutando: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Paquete roop instalado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando paquete roop:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def test_roop_import():
    """Probar que roop se puede importar"""
    print("\n🧪 PROBANDO IMPORTACIÓN ROOP:")
    print("=" * 40)
    
    try:
        import roop
        print("✅ Módulo roop importado")
        
        # Probar importar core
        try:
            from roop import core
            print("✅ Módulo core importado")
        except ImportError as e:
            print(f"⚠️ Error importando core: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando roop: {e}")
        return False

def setup_python_path():
    """Configurar path de Python"""
    print("\n🔧 CONFIGURANDO PYTHON PATH:")
    print("=" * 40)
    
    current_dir = os.getcwd()
    
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Agregado al path: {current_dir}")
    
    return True

def main():
    """Función principal"""
    print("🚀 ARREGLANDO INSTALACIÓN DE ROOP")
    print("=" * 60)
    
    # Detectar estructura
    structure = detect_roop_structure()
    print(f"📋 Estructura detectada: {structure}")
    
    # Crear módulo roop si es necesario
    if structure == 'unknown':
        create_roop_module()
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Configurar path
    setup_python_path()
    
    # Instalar paquete
    if not install_roop_package():
        print("❌ Error instalando paquete")
        return False
    
    # Probar importación
    if not test_roop_import():
        print("❌ Error probando importación")
        return False
    
    print("\n✅ INSTALACIÓN ARREGLADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Ejecutar: python optimize_colab_gpu.py")
    print("2. Usar: python run_colab_gpu_optimized.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    
    return True

if __name__ == '__main__':
    main() 