#!/usr/bin/env python3
"""
Script para solucionar el conflicto de versiones de NumPy
"""

import os
import sys
import subprocess

def fix_numpy_version():
    """Solucionar conflicto de versiones de NumPy"""
    print("🔧 SOLUCIONANDO CONFLICTO DE VERSIONES NUMPY")
    print("=" * 50)
    
    # Verificar numpy actual
    try:
        import numpy as np
        print(f"NumPy version actual: {np.__version__}")
        
        if np.__version__.startswith('2.'):
            print("❌ NumPy 2.x detectado - incompatible con las librerías")
            print("📦 Downgradeando a NumPy 1.x...")
            
            # Desinstalar numpy actual
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"], 
                          capture_output=True, text=True)
            
            # Instalar numpy 1.x compatible
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "numpy==1.26.4"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ NumPy downgradeado exitosamente")
                
                # Verificar nueva versión
                try:
                    import numpy as np
                    print(f"✅ Nueva versión NumPy: {np.__version__}")
                    return True
                except ImportError:
                    print("❌ Error importando NumPy después del downgrade")
                    return False
            else:
                print(f"❌ Error downgradeando NumPy: {result.stderr}")
                return False
        else:
            print("✅ NumPy versión compatible")
            return True
            
    except ImportError:
        print("❌ NumPy no instalado")
        return False

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
    print("🚀 SOLUCIONADOR CONFLICTO NUMPY")
    print("=" * 50)
    
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
    response = input("\n¿Proceder con la corrección? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        if fix_numpy_version():
            test_libraries()
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