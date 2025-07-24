#!/usr/bin/env python3
"""
Script final para forzar el downgrade de NumPy a versión 1.x
"""

import os
import sys
import subprocess
import shutil

def force_numpy_downgrade():
    """Forzar downgrade de NumPy a versión 1.x"""
    print("🚀 FORZANDO DOWNGRADE DE NUMPY A VERSIÓN 1.X")
    print("=" * 60)
    
    # Paso 1: Desinstalar NumPy completamente
    print("🗑️ Paso 1: Desinstalando NumPy completamente...")
    
    commands = [
        [sys.executable, "-m", "pip", "uninstall", "numpy", "-y"],
        [sys.executable, "-m", "pip", "cache", "purge"],
        ["rm", "-rf", "/root/.cache/pip"],
        ["rm", "-rf", "/usr/local/lib/python3.11/dist-packages/numpy*"],
        ["rm", "-rf", "/usr/lib/python3.11/dist-packages/numpy*"]
    ]
    
    for cmd in commands:
        try:
            print(f"Ejecutando: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Comando exitoso")
            else:
                print(f"⚠️ Comando falló (puede ser normal): {result.stderr}")
        except Exception as e:
            print(f"⚠️ Error (puede ser normal): {e}")
    
    # Paso 2: Limpiar cache de pip
    print("\n🧹 Paso 2: Limpiando cache de pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=True)
        print("✅ Cache limpiado")
    except:
        print("⚠️ Error limpiando cache")
    
    # Paso 3: Instalar NumPy 1.26.4 específicamente
    print("\n📦 Paso 3: Instalando NumPy 1.26.4...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.26.4", 
            "--no-cache-dir", 
            "--force-reinstall"
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
    print("\n🔍 Paso 4: Verificando instalación...")
    try:
        import numpy as np
        print(f"✅ NumPy instalado: {np.__version__}")
        
        # Verificar módulos críticos
        try:
            import numpy.typing
            print("✅ numpy.typing disponible")
        except ImportError:
            print("❌ numpy.typing no disponible")
            return False
            
        try:
            import numpy.strings
            print("✅ numpy.strings disponible")
        except ImportError:
            print("⚠️ numpy.strings no disponible (puede ser normal)")
            
        return True
        
    except ImportError as e:
        print(f"❌ Error importando NumPy: {e}")
        return False

def reinstall_critical_packages():
    """Reinstalar paquetes críticos"""
    print("\n📦 REINSTALANDO PAQUETES CRÍTICOS")
    print("=" * 60)
    
    packages = [
        "opencv-python",
        "tensorflow==2.15.0",
        "torch",
        "insightface",
        "onnxruntime-gpu==1.15.1"
    ]
    
    for package in packages:
        print(f"📦 Reinstalando {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                package, 
                "--no-cache-dir", 
                "--force-reinstall"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} reinstalado")
            else:
                print(f"⚠️ Error con {package}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_imports():
    """Probar imports críticos"""
    print("\n🧪 PROBANDO IMPORTS CRÍTICOS")
    print("=" * 60)
    
    imports = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("onnxruntime", "ONNX Runtime"),
        ("insightface", "InsightFace")
    ]
    
    success_count = 0
    
    for module_name, display_name in imports:
        try:
            if module_name == "numpy":
                import numpy as np
                print(f"✅ {display_name}: {np.__version__}")
            elif module_name == "cv2":
                import cv2
                print(f"✅ {display_name}: {cv2.__version__}")
            elif module_name == "torch":
                import torch
                print(f"✅ {display_name}: {torch.__version__}")
                print(f"   CUDA: {torch.cuda.is_available()}")
            elif module_name == "tensorflow":
                import tensorflow as tf
                print(f"✅ {display_name}: {tf.__version__}")
            elif module_name == "onnxruntime":
                import onnxruntime as ort
                print(f"✅ {display_name}: {ort.__version__}")
                print(f"   Providers: {ort.get_available_providers()}")
            elif module_name == "insightface":
                import insightface
                print(f"✅ {display_name}: {insightface.__version__}")
            else:
                __import__(module_name)
                print(f"✅ {display_name}: disponible")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {display_name}: {e}")
    
    return success_count == len(imports)

def test_face_swapper():
    """Probar face swapper"""
    print("\n🎭 PROBANDO FACE SWAPPER")
    print("=" * 60)
    
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
            return True
        else:
            print("❌ Error cargando face swapper")
            return False
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
        return False

def main():
    print("🚀 SOLUCIONADOR FINAL DE NUMPY")
    print("=" * 60)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        if np.__version__.startswith('2'):
            print("⚠️ NumPy 2.x detectado - necesita downgrade")
        else:
            print("✅ NumPy 1.x detectado")
    except ImportError:
        print("NumPy: No instalado")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con el downgrade forzado? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        # Forzar downgrade
        if force_numpy_downgrade():
            print("\n✅ Downgrade exitoso")
            
            # Reinstalar paquetes críticos
            reinstall_critical_packages()
            
            # Probar imports
            if test_imports():
                print("\n✅ Todos los imports funcionan")
                
                # Probar face swapper
                if test_face_swapper():
                    print("\n🎉 ¡TODO FUNCIONA PERFECTAMENTE!")
                    print("=" * 60)
                    print("Ahora puedes ejecutar:")
                    print("python test_gpu_force.py")
                    print()
                    print("Y luego el procesamiento por lotes:")
                    print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/113.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps")
                else:
                    print("\n⚠️ Face swapper no funciona completamente")
            else:
                print("\n❌ Algunos imports fallan")
        else:
            print("\n❌ Downgrade falló")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 