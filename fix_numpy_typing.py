#!/usr/bin/env python3
"""
Script para solucionar el problema de numpy.typing
"""

import os
import sys
import subprocess

def fix_numpy_typing():
    """Solucionar problema de numpy.typing"""
    print("🔧 SOLUCIONANDO PROBLEMA NUMPY.TYPING")
    print("=" * 50)
    
    # Verificar numpy actual
    try:
        import numpy as np
        print(f"NumPy version actual: {np.__version__}")
    except ImportError:
        print("❌ NumPy no instalado")
        return False
    
    # Verificar numpy.typing
    try:
        import numpy.typing
        print("✅ numpy.typing disponible")
        return True
    except ImportError as e:
        print(f"❌ Error numpy.typing: {e}")
    
    # Solucionar problema
    print("📦 Actualizando NumPy...")
    try:
        # Desinstalar numpy actual
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], 
                      capture_output=True, text=True)
        
        # Instalar numpy versión compatible
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "numpy>=1.26.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ NumPy actualizado exitosamente")
            
            # Verificar nuevamente
            try:
                import numpy.typing
                print("✅ numpy.typing ahora disponible")
                return True
            except ImportError:
                print("❌ numpy.typing aún no disponible")
                return False
        else:
            print(f"❌ Error actualizando NumPy: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_face_swapper():
    """Probar face swapper después de la corrección"""
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
    print("🚀 SOLUCIONADOR NUMPY.TYPING")
    print("=" * 50)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: No instalado")
    
    try:
        import numpy.typing
        print("numpy.typing: ✅ Disponible")
    except ImportError:
        print("numpy.typing: ❌ No disponible")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con la corrección? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        if fix_numpy_typing():
            test_face_swapper()
            
            print("\n🎉 PROCESO COMPLETADO")
            print("=" * 50)
            print("Ahora puedes ejecutar:")
            print("python test_gpu_force.py")
        else:
            print("\n❌ No se pudo solucionar el problema")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 