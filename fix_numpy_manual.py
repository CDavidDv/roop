#!/usr/bin/env python3
"""
Script manual para solucionar el problema de NumPy
"""

import os
import sys
import subprocess

def manual_numpy_fix():
    """Solución manual para NumPy"""
    print("🔧 SOLUCIÓN MANUAL NUMPY")
    print("=" * 50)
    
    print("Ejecutando comandos manuales...")
    
    # Comando 1: Desinstalar numpy
    print("\n📦 Comando 1: Desinstalando NumPy...")
    cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "numpy"]
    print(f"Ejecutando: {' '.join(cmd1)}")
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    print(f"Resultado: {result1.returncode}")
    
    # Comando 2: Limpiar caché
    print("\n📦 Comando 2: Limpiando caché...")
    cmd2 = [sys.executable, "-m", "pip", "cache", "purge"]
    print(f"Ejecutando: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    print(f"Resultado: {result2.returncode}")
    
    # Comando 3: Instalar numpy específico
    print("\n📦 Comando 3: Instalando NumPy 1.26.4...")
    cmd3 = [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--no-cache-dir"]
    print(f"Ejecutando: {' '.join(cmd3)}")
    result3 = subprocess.run(cmd3, capture_output=True, text=True)
    print(f"Resultado: {result3.returncode}")
    
    if result3.returncode == 0:
        print("✅ Instalación exitosa")
        return True
    else:
        print(f"❌ Error: {result3.stderr}")
        return False

def verify_installation():
    """Verificar la instalación"""
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 50)
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        if np.__version__.startswith('1.'):
            print("✅ Versión compatible")
            
            # Probar módulos
            try:
                import numpy.typing
                print("✅ numpy.typing OK")
            except:
                print("⚠️ numpy.typing no disponible")
            
            try:
                import numpy.strings
                print("✅ numpy.strings OK")
            except:
                print("⚠️ numpy.strings no disponible")
            
            return True
        else:
            print(f"❌ Versión incorrecta: {np.__version__}")
            return False
            
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

def test_face_swapper():
    """Probar face swapper"""
    print("\n🎭 PROBANDO FACE SWAPPER")
    print("=" * 50)
    
    try:
        import roop.processors.frame.face_swapper as face_swapper
        
        print("Cargando modelo de face swapper...")
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            
            # Verificar proveedores
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
                if 'CUDAExecutionProvider' in swapper.providers:
                    print("✅ Face swapper usando GPU")
                else:
                    print("⚠️ Face swapper usando CPU")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")

def main():
    print("🚀 SOLUCIÓN MANUAL NUMPY")
    print("=" * 50)
    
    # Mostrar comandos que se ejecutarán
    print("Comandos que se ejecutarán:")
    print("1. pip uninstall numpy -y")
    print("2. pip cache purge")
    print("3. pip install numpy==1.26.4 --no-cache-dir")
    
    # Preguntar si proceder
    response = input("\n¿Proceder? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        if manual_numpy_fix():
            if verify_installation():
                test_face_swapper()
                
                print("\n🎉 PROCESO COMPLETADO")
                print("=" * 50)
                print("Ahora puedes ejecutar:")
                print("python test_gpu_force.py")
            else:
                print("\n❌ Verificación falló")
        else:
            print("\n❌ Instalación falló")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 