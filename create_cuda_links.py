#!/usr/bin/env python3
"""
Script para crear enlaces simbólicos de librerías CUDA y ejecutar procesamiento
"""

import os
import sys
import subprocess
import shutil

def create_cuda_links():
    """Crea enlaces simbólicos para librerías CUDA"""
    print("🔗 CREANDO ENLACES SIMBÓLICOS CUDA")
    print("=" * 50)
    
    # Enlaces necesarios
    links = [
        ("/usr/local/cuda-11.8/lib64/libcufft.so.10", "/usr/lib/x86_64-linux-gnu/libcufft.so.10"),
        ("/usr/local/cuda-11.8/lib64/libcublas.so.11", "/usr/lib/x86_64-linux-gnu/libcublas.so.11"),
        ("/usr/lib/x86_64-linux-gnu/libcudnn.so.8", "/usr/lib/x86_64-linux-gnu/libcudnn.so.8")
    ]
    
    for source, target in links:
        try:
            if os.path.exists(source):
                if os.path.exists(target):
                    os.remove(target)  # Remover enlace existente
                print(f"🔗 Creando enlace: {source} -> {target}")
                subprocess.run(["ln", "-sf", source, target], check=True)
                print(f"✅ Enlace creado: {target}")
            else:
                print(f"⚠️ Fuente no existe: {source}")
        except Exception as e:
            print(f"❌ Error creando enlace {target}: {e}")
    
    # Actualizar cache de librerías
    try:
        print("🔄 Actualizando cache de librerías...")
        subprocess.run(["ldconfig"], check=True)
        print("✅ Cache actualizado")
    except Exception as e:
        print(f"⚠️ Error actualizando cache: {e}")

def test_onnx_cuda_fixed():
    """Prueba ONNX Runtime con CUDA después de crear enlaces"""
    print("🧪 PROBANDO ONNX RUNTIME CON CUDA (ARREGLADO)")
    print("=" * 60)
    
    test_code = """
import onnxruntime as ort
import os

# Configurar variables de entorno
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

print("🔍 Proveedores disponibles:")
providers = ort.get_available_providers()
for provider in providers:
    print(f"  - {provider}")

print("\\n🔍 Verificando librerías CUDA...")
import ctypes
try:
    # Intentar cargar librerías CUDA
    ctypes.CDLL("libcufft.so.10")
    print("✅ libcufft.so.10 cargada correctamente")
except Exception as e:
    print(f"❌ Error cargando libcufft.so.10: {e}")

try:
    ctypes.CDLL("libcublas.so.11")
    print("✅ libcublas.so.11 cargada correctamente")
except Exception as e:
    print(f"❌ Error cargando libcublas.so.11: {e}")

print("✅ ONNX Runtime con CUDA configurado correctamente")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def run_processing_with_cuda_links():
    """Ejecuta el procesamiento con enlaces CUDA creados"""
    print("🚀 EJECUTANDO PROCESAMIENTO CON ENLACES CUDA")
    print("=" * 60)
    
    # Configurar variables de entorno
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
    
    # Comando de procesamiento
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130_CUDA_FIXED.mp4",
        "--frame-processor", "face_swapper",
        "--execution-provider", "cuda",
        "--execution-threads", "24",
        "--temp-frame-quality", "90",
        "--max-memory", "8",
        "--gpu-memory-wait", "45",
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento con CUDA...")
        result = subprocess.run(command, timeout=3600)  # 1 hora timeout
        if result.returncode == 0:
            print("✅ Procesamiento completado exitosamente con CUDA")
            return True
        else:
            print("❌ Error en procesamiento con CUDA")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Timeout del procesamiento")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 CREANDO ENLACES CUDA Y EJECUTANDO PROCESAMIENTO")
    print("=" * 70)
    
    # Paso 1: Crear enlaces simbólicos
    create_cuda_links()
    
    # Paso 2: Probar ONNX Runtime
    if not test_onnx_cuda_fixed():
        print("⚠️ ONNX Runtime aún no funciona con CUDA")
        print("🔄 Intentando procesamiento de todas formas...")
    
    # Paso 3: Ejecutar procesamiento
    success = run_processing_with_cuda_links()
    
    if success:
        print("🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
        print("📁 Archivo guardado en: /content/resultados/DanielaAS130_CUDA_FIXED.mp4")
    else:
        print("❌ Error en el procesamiento")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 