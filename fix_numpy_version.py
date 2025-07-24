#!/usr/bin/env python3
"""
Script para solucionar el conflicto de versiones de NumPy
"""

import os
import sys
import subprocess

def force_numpy_downgrade():
    """Forzar downgrade de NumPy de manera más agresiva"""
    print("🔧 FORZANDO DOWNGRADE DE NUMPY")
    print("=" * 50)
    
    # Paso 1: Desinstalar numpy completamente
    print("📦 Paso 1: Desinstalando NumPy completamente...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], 
                      capture_output=True, text=True)
        print("✅ NumPy desinstalado")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    # Paso 2: Limpiar caché de pip
    print("📦 Paso 2: Limpiando caché de pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      capture_output=True, text=True)
        print("✅ Caché limpiado")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    # Paso 3: Instalar numpy 1.26.4 específicamente
    print("📦 Paso 3: Instalando NumPy 1.26.4...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.26.4", "--no-cache-dir", "--force-reinstall"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ NumPy 1.26.4 instalado")
        else:
            print(f"❌ Error instalando NumPy: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Paso 4: Verificar instalación
    print("📦 Paso 4: Verificando instalación...")
    try:
        import numpy as np
        print(f"✅ NumPy instalado: {np.__version__}")
        
        if np.__version__.startswith('1.'):
            print("✅ Versión compatible instalada")
            return True
        else:
            print(f"❌ Versión incorrecta: {np.__version__}")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando NumPy: {e}")
        return False

def test_numpy_modules():
    """Probar módulos de NumPy"""
    print("\n🧪 PROBANDO MÓDULOS NUMPY")
    print("=" * 50)
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        # Probar numpy.typing
        try:
            import numpy.typing
            print("✅ numpy.typing disponible")
        except ImportError as e:
            print(f"❌ numpy.typing: {e}")
        
        # Probar numpy.strings
        try:
            import numpy.strings
            print("✅ numpy.strings disponible")
        except ImportError as e:
            print(f"❌ numpy.strings: {e}")
        
        # Probar numpy.core
        try:
            import numpy.core
            print("✅ numpy.core disponible")
        except ImportError as e:
            print(f"❌ numpy.core: {e}")
            
    except ImportError as e:
        print(f"❌ Error NumPy: {e}")

def test_libraries():
    """Probar que las librerías funcionan"""
    print("\n🧪 PROBANDO LIBRERÍAS")
    print("=" * 50)
    
    # Probar ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime: {ort.__version__}")
        print(f"   Providers: {ort.get_available_providers()}")
    except Exception as e:
        print(f"❌ Error ONNX Runtime: {e}")
    
    # Probar PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")
    
    # Probar OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"❌ Error OpenCV: {e}")
    
    # Probar insightface
    try:
        import insightface
        print(f"✅ InsightFace: {insightface.__version__}")
    except Exception as e:
        print(f"❌ Error InsightFace: {e}")

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
    print("🚀 SOLUCIONADOR CONFLICTO NUMPY - VERSIÓN MEJORADA")
    print("=" * 60)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        if np.__version__.startswith('2.'):
            print("⚠️ NumPy 2.x detectado - necesita downgrade")
        else:
            print("✅ NumPy versión compatible")
    except ImportError:
        print("NumPy: No instalado")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con la corrección forzada? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        if force_numpy_downgrade():
            test_numpy_modules()
            test_libraries()
            test_face_swapper()
            
            print("\n🎉 PROCESO COMPLETADO")
            print("=" * 50)
            print("Ahora puedes ejecutar:")
            print("python test_gpu_force.py")
        else:
            print("\n❌ No se pudo solucionar el problema")
            print("Intenta manualmente:")
            print("pip uninstall numpy -y")
            print("pip install numpy==1.26.4 --no-cache-dir")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 