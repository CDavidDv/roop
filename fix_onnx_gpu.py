#!/usr/bin/env python3
"""
Script específico para solucionar problemas de ONNX Runtime GPU en Google Colab
"""

import os
import sys
import subprocess

def fix_onnx_gpu():
    """Solucionar problema de ONNX Runtime GPU"""
    print("🔧 SOLUCIONANDO PROBLEMA ONNX RUNTIME GPU")
    print("=" * 50)
    
    # Paso 1: Desinstalar onnxruntime CPU
    print("📦 Paso 1: Desinstalando ONNX Runtime CPU...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], 
                      capture_output=True, text=True)
        print("✅ ONNX Runtime CPU desinstalado")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    # Paso 2: Instalar numpy.typing si falta
    print("📦 Paso 2: Verificando numpy.typing...")
    try:
        import numpy.typing
        print("✅ numpy.typing disponible")
    except ImportError:
        print("📦 Instalando numpy.typing...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "numpy>=1.26.0"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ numpy actualizado")
            else:
                print(f"⚠️ Error actualizando numpy: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    # Paso 3: Instalar onnxruntime-gpu específico para Colab
    print("📦 Paso 3: Instalando ONNX Runtime GPU...")
    try:
        # Forzar desinstalación completa
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"], 
                      capture_output=True, text=True)
        
        # Instalar versión específica para Colab
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "onnxruntime-gpu==1.15.1", "--no-cache-dir", "--force-reinstall"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ONNX Runtime GPU instalado exitosamente")
        else:
            print(f"❌ Error: {result.stderr}")
            # Intentar con versión alternativa
            print("📦 Intentando con versión alternativa...")
            result2 = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "onnxruntime-gpu", "--no-cache-dir", "--force-reinstall"
            ], capture_output=True, text=True)
            
            if result2.returncode == 0:
                print("✅ ONNX Runtime GPU instalado (versión alternativa)")
            else:
                print(f"❌ Error con versión alternativa: {result2.stderr}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Paso 4: Verificar instalación
    print("📦 Paso 4: Verificando instalación...")
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"ONNX Runtime file: {ort.__file__}")
        
        # Verificar si es GPU o CPU de manera más precisa
        ort_path = ort.__file__
        if 'onnxruntime-gpu' in ort_path or 'gpu' in ort_path.lower():
            print("✅ ONNX Runtime GPU instalado correctamente")
        else:
            print("❌ ONNX Runtime CPU aún instalado")
            print(f"   Ruta: {ort_path}")
            
        providers = ort.get_available_providers()
        print(f"Providers disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible")
            
            # Probar crear una sesión con CUDA
            try:
                import numpy as np
                # Crear un modelo simple para probar
                import onnx
                from onnx import helper
                
                # Crear un modelo ONNX simple
                X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
                Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
                
                node = helper.make_node('Identity', ['X'], ['Y'])
                graph = helper.make_graph([node], 'test', [X], [Y])
                model = helper.make_model(graph)
                
                # Probar con CUDA
                session = ort.InferenceSession(model.SerializeToString(), 
                                             providers=['CUDAExecutionProvider'])
                print("✅ Sesión CUDA funcionando correctamente")
                
            except Exception as e:
                print(f"⚠️ Error en sesión CUDA: {e}")
        else:
            print("❌ CUDA GPU no disponible")
            
    except ImportError as e:
        print(f"❌ Error importando ONNX Runtime: {e}")

def test_face_swapper():
    """Probar face swapper después de la instalación"""
    print("\n🎭 PROBANDO FACE SWAPPER")
    print("=" * 50)
    
    try:
        # Verificar numpy.typing primero
        try:
            import numpy.typing
            print("✅ numpy.typing disponible")
        except ImportError as e:
            print(f"❌ Error numpy.typing: {e}")
            return
        
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
        import traceback
        traceback.print_exc()

def main():
    print("🚀 SOLUCIONADOR ONNX RUNTIME GPU - GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"Archivo: {ort.__file__}")
        print(f"Providers: {ort.get_available_providers()}")
    except ImportError:
        print("ONNX Runtime no instalado")
    
    # Verificar numpy.typing
    try:
        import numpy.typing
        print("✅ numpy.typing disponible")
    except ImportError:
        print("❌ numpy.typing no disponible")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con la instalación? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        fix_onnx_gpu()
        test_face_swapper()
        
        print("\n🎉 PROCESO COMPLETADO")
        print("=" * 50)
        print("Si el problema persiste, ejecuta:")
        print("python test_gpu_force.py")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 