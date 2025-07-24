#!/usr/bin/env python3
"""
Script para solucionar los problemas restantes del entorno
"""

import os
import sys
import subprocess

def fix_remaining_issues():
    """Solucionar problemas restantes"""
    print("🔧 SOLUCIONANDO PROBLEMAS RESTANTES")
    print("=" * 50)
    
    # Paso 1: Instalar dependencias faltantes
    print("📦 Paso 1: Instalando dependencias faltantes...")
    
    dependencies = [
        "sympy",
        "onnx",
        "opencv-python",
        "insightface",
        "gfpgan",
        "basicsr",
        "facexlib"
    ]
    
    for dep in dependencies:
        print(f"📦 Instalando {dep}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {dep} instalado")
            else:
                print(f"⚠️ Error con {dep}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Paso 2: Verificar protobuf
    print("\n📦 Paso 2: Verificando protobuf...")
    try:
        import google.protobuf
        print(f"✅ protobuf: {google.protobuf.__version__}")
    except ImportError:
        print("📦 Instalando protobuf...")
        subprocess.run([sys.executable, "-m", "pip", "install", "protobuf==4.23.4"])
    
    # Paso 3: Verificar flatbuffers
    print("\n📦 Paso 3: Verificando flatbuffers...")
    try:
        import flatbuffers
        print(f"✅ flatbuffers disponible")
    except ImportError:
        print("📦 Instalando flatbuffers...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flatbuffers>=23.5.26"])

def test_all_libraries():
    """Probar todas las librerías"""
    print("\n🧪 PROBANDO TODAS LAS LIBRERÍAS")
    print("=" * 50)
    
    libraries = [
        ("numpy", "NumPy"),
        ("onnxruntime", "ONNX Runtime"),
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("insightface", "InsightFace"),
        ("sympy", "SymPy"),
        ("onnx", "ONNX"),
        ("flatbuffers", "FlatBuffers"),
        ("google.protobuf", "Protobuf")
    ]
    
    for module_name, display_name in libraries:
        try:
            if module_name == "cv2":
                import cv2
                print(f"✅ {display_name}: {cv2.__version__}")
            elif module_name == "torch":
                import torch
                print(f"✅ {display_name}: {torch.__version__}")
                print(f"   CUDA: {torch.cuda.is_available()}")
            elif module_name == "tensorflow":
                import tensorflow as tf
                print(f"✅ {display_name}: {tf.__version__}")
            elif module_name == "numpy":
                import numpy as np
                print(f"✅ {display_name}: {np.__version__}")
            elif module_name == "onnxruntime":
                import onnxruntime as ort
                print(f"✅ {display_name}: {ort.__version__}")
                print(f"   Providers: {ort.get_available_providers()}")
            elif module_name == "insightface":
                import insightface
                print(f"✅ {display_name}: {insightface.__version__}")
            elif module_name == "sympy":
                import sympy
                print(f"✅ {display_name}: {sympy.__version__}")
            elif module_name == "onnx":
                import onnx
                print(f"✅ {display_name}: {onnx.__version__}")
            elif module_name == "flatbuffers":
                import flatbuffers
                print(f"✅ {display_name}: disponible")
            elif module_name == "google.protobuf":
                import google.protobuf
                print(f"✅ {display_name}: {google.protobuf.__version__}")
            else:
                __import__(module_name)
                print(f"✅ {display_name}: disponible")
        except Exception as e:
            print(f"❌ {display_name}: {e}")

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

def test_face_enhancer():
    """Probar face enhancer"""
    print("\n✨ PROBANDO FACE ENHANCER")
    print("=" * 50)
    
    try:
        import roop.processors.frame.face_enhancer as face_enhancer
        
        device = face_enhancer.get_device()
        print(f"Dispositivo detectado: {device}")
        
        if device == 'cuda':
            print("✅ Face enhancer configurado para usar GPU")
        else:
            print(f"⚠️ Face enhancer usando: {device}")
            
    except Exception as e:
        print(f"❌ Error probando face enhancer: {e}")

def main():
    print("🚀 SOLUCIONADOR PROBLEMAS RESTANTES")
    print("=" * 50)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: No instalado")
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"Providers: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime: No instalado")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con la corrección? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        fix_remaining_issues()
        test_all_libraries()
        test_face_swapper()
        test_face_enhancer()
        
        print("\n🎉 PROCESO COMPLETADO")
        print("=" * 50)
        print("Ahora puedes ejecutar:")
        print("python test_gpu_force.py")
        print()
        print("Y luego el procesamiento por lotes:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/113.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 