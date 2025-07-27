#!/usr/bin/env python3
"""
Script para arreglar la estructura de directorios anidados
Detecta y navega al directorio correcto donde está el módulo roop
"""

import os
import sys
import subprocess

def find_roop_module():
    """Buscar el directorio que contiene el módulo roop"""
    print("🔍 BUSCANDO MÓDULO ROOP:")
    print("=" * 40)
    
    current_dir = os.getcwd()
    print(f"📁 Directorio actual: {current_dir}")
    
    # Buscar recursivamente el módulo roop
    def search_roop_module(start_path):
        for root, dirs, files in os.walk(start_path):
            # Verificar si hay un directorio 'roop' con __init__.py
            if 'roop' in dirs:
                roop_path = os.path.join(root, 'roop')
                init_file = os.path.join(roop_path, '__init__.py')
                if os.path.exists(init_file):
                    return roop_path
            # Verificar si hay run.py en este directorio
            if 'run.py' in files:
                run_path = os.path.join(root, 'run.py')
                # Verificar si este run.py importa roop
                try:
                    with open(run_path, 'r') as f:
                        content = f.read()
                        if 'from roop import' in content:
                            return root
                except:
                    pass
        return None
    
    # Buscar desde el directorio actual
    roop_dir = search_roop_module(current_dir)
    
    if roop_dir:
        print(f"✅ Módulo roop encontrado en: {roop_dir}")
        return roop_dir
    else:
        print("❌ Módulo roop no encontrado")
        return None

def navigate_to_correct_directory():
    """Navegar al directorio correcto"""
    print("\n📂 NAVEGANDO AL DIRECTORIO CORRECTO:")
    print("=" * 40)
    
    roop_dir = find_roop_module()
    
    if not roop_dir:
        print("❌ No se pudo encontrar el módulo roop")
        return False
    
    # Cambiar al directorio correcto
    os.chdir(roop_dir)
    print(f"✅ Cambiado a: {os.getcwd()}")
    
    # Verificar que estamos en el lugar correcto
    if os.path.exists('run.py'):
        print("✅ run.py encontrado")
    else:
        print("❌ run.py no encontrado")
        return False
    
    # Verificar que el módulo roop está disponible
    try:
        import roop
        print("✅ Módulo roop importado exitosamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando roop: {e}")
        return False

def setup_python_path():
    """Configurar el path de Python"""
    print("\n🔧 CONFIGURANDO PYTHON PATH:")
    print("=" * 40)
    
    current_dir = os.getcwd()
    
    # Agregar el directorio actual al path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"✅ Agregado al path: {current_dir}")
    
    # Agregar el directorio padre al path (por si el módulo está ahí)
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"✅ Agregado al path: {parent_dir}")
    
    return True

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

def create_directory_fix_script():
    """Crear script que arregle automáticamente el directorio"""
    print("\n📝 CREANDO SCRIPT DE ARREGLO:")
    print("=" * 40)
    
    script_content = '''#!/usr/bin/env python3
"""
Script para arreglar automáticamente el directorio de ROOP
"""

import os
import sys
import subprocess

def find_and_navigate_to_roop():
    """Buscar y navegar al directorio correcto de ROOP"""
    print("🔍 BUSCANDO DIRECTORIO CORRECTO DE ROOP...")
    
    current_dir = os.getcwd()
    
    # Buscar recursivamente
    def search_roop(start_path):
        for root, dirs, files in os.walk(start_path):
            if 'run.py' in files:
                run_path = os.path.join(root, 'run.py')
                try:
                    with open(run_path, 'r') as f:
                        content = f.read()
                        if 'from roop import' in content:
                            return root
                except:
                    pass
        return None
    
    roop_dir = search_roop(current_dir)
    
    if roop_dir:
        print(f"✅ Directorio encontrado: {roop_dir}")
        os.chdir(roop_dir)
        print(f"✅ Cambiado a: {os.getcwd()}")
        return True
    else:
        print("❌ Directorio no encontrado")
        return False

def setup_environment():
    """Configurar entorno"""
    print("⚙️ CONFIGURANDO ENTORNO...")
    
    # Variables de entorno para GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Configurar path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print("✅ Entorno configurado")

def main():
    """Función principal"""
    print("🚀 ARREGLANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 50)
    
    # Buscar y navegar al directorio correcto
    if not find_and_navigate_to_roop():
        print("❌ No se pudo encontrar el directorio correcto")
        return False
    
    # Configurar entorno
    setup_environment()
    
    # Probar importación
    try:
        import roop
        print("✅ ROOP funcionando correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    main()
'''
    
    try:
        with open('fix_directory.py', 'w') as f:
            f.write(script_content)
        print("✅ Script de arreglo creado: fix_directory.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 60)
    
    # Navegar al directorio correcto
    if not navigate_to_correct_directory():
        print("❌ No se pudo navegar al directorio correcto")
        return False
    
    # Configurar path
    setup_python_path()
    
    # Probar importación
    if not test_roop_import():
        print("❌ Error probando importación")
        return False
    
    # Crear script de arreglo
    create_directory_fix_script()
    
    print("\n✅ ESTRUCTURA ARREGLADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Ejecutar: python fix_directory.py")
    print("2. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 