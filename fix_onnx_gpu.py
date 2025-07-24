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
    
    # Paso 2: Instalar onnxruntime-gpu específico para Colab
    print("📦 Paso 2: Instalando ONNX Runtime GPU...")
    try:
        # Intentar con versión específica para Colab
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "onnxruntime-gpu==1.15.1", "--force-reinstall"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ONNX Runtime GPU instalado exitosamente")
        else:
            print(f"❌ Error: {result.stderr}")
            # Intentar con versión alternativa
            print("📦 Intentando con versión alternativa...")
            result2 = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "onnxruntime-gpu", "--force-reinstall"
            ], capture_output=True, text=True)
            
            if result2.returncode == 0:
                print("✅ ONNX Runtime GPU instalado (versión alternativa)")
            else:
                print(f"❌ Error con versión alternativa: {result2.stderr}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Paso 3: Verificar instalación
    print("📦 Paso 3: Verificando instalación...")
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"ONNX Runtime file: {ort.__file__}")
        
        if 'onnxruntime-gpu' in ort.__file__:
            print("✅ ONNX Runtime GPU instalado correctamente")
        else:
            print("❌ ONNX Runtime CPU aún instalado")
            
        providers = ort.get_available_providers()
        print(f"Providers disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible")
        else:
            print("❌ CUDA GPU no disponible")
            
    except ImportError as e:
        print(f"❌ Error importando ONNX Runtime: {e}")

def test_face_swapper():
    """Probar face swapper después de la instalación"""
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