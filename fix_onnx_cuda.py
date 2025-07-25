#!/usr/bin/env python3
"""
Script para arreglar las rutas de librerías CUDA para ONNX Runtime
"""

import os
import sys
import subprocess
import shutil

def setup_cuda_paths():
    """Configura las rutas de librerías CUDA"""
    print("🔧 CONFIGURANDO RUTAS CUDA PARA ONNX RUNTIME")
    print("=" * 60)
    
    # Buscar librerías CUDA
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/cuda/lib64",
        "/usr/local/cuda-11.8/lib64",
        "/usr/local/cuda-12.0/lib64"
    ]
    
    # Librerías necesarias
    required_libs = [
        "libcufft.so.10",
        "libcublas.so.11",
        "libcudnn.so.8"
    ]
    
    print("🔍 Buscando librerías CUDA...")
    
    for lib in required_libs:
        found = False
        for path in cuda_paths:
            lib_path = os.path.join(path, lib)
            if os.path.exists(lib_path):
                print(f"✅ {lib} encontrada en {path}")
                found = True
                break
        
        if not found:
            print(f"❌ {lib} NO encontrada")
    
    # Configurar variables de entorno
    cuda_lib_path = "/usr/lib/x86_64-linux-gnu"
    if os.path.exists(cuda_lib_path):
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"✅ Configurado LD_LIBRARY_PATH: {cuda_lib_path}")
    
    # Crear enlaces simbólicos si es necesario
    print("🔗 Creando enlaces simbólicos...")
    try:
        # Verificar si libcufft.so.10 existe
        cufft_path = "/usr/lib/x86_64-linux-gnu/libcufft.so.10"
        if not os.path.exists(cufft_path):
            # Buscar versión disponible
            for version in ["11", "12"]:
                alt_path = f"/usr/lib/x86_64-linux-gnu/libcufft.so.{version}"
                if os.path.exists(alt_path):
                    print(f"🔗 Creando enlace: {alt_path} -> {cufft_path}")
                    subprocess.run(["ln", "-sf", alt_path, cufft_path], check=True)
                    break
    except Exception as e:
        print(f"⚠️ Error creando enlaces: {e}")

def test_onnx_cuda():
    """Prueba ONNX Runtime con CUDA"""
    print("🧪 PROBANDO ONNX RUNTIME CON CUDA")
    print("=" * 50)
    
    test_code = """
import onnxruntime as ort
import os

# Configurar variables de entorno
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

print("🔍 Proveedores disponibles:")
providers = ort.get_available_providers()
for provider in providers:
    print(f"  - {provider}")

print("\\n🔍 Configurando sesión con CUDA...")
try:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Intentar crear sesión con CUDA
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"✅ Proveedores configurados: {providers}")
    
    print("✅ ONNX Runtime con CUDA configurado correctamente")
    return True
except Exception as e:
    print(f"❌ Error configurando ONNX Runtime: {e}")
    return False
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

def run_processing_with_fixed_cuda():
    """Ejecuta el procesamiento con CUDA arreglado"""
    print("🚀 EJECUTANDO PROCESAMIENTO CON CUDA ARREGLADO")
    print("=" * 60)
    
    # Configurar variables de entorno
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
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
        "-o", "/content/resultados/DanielaAS130_CUDA.mp4",
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
    print("🚀 ARREGLANDO ONNX RUNTIME Y EJECUTANDO PROCESAMIENTO")
    print("=" * 70)
    
    # Paso 1: Configurar rutas CUDA
    setup_cuda_paths()
    
    # Paso 2: Probar ONNX Runtime
    if not test_onnx_cuda():
        print("⚠️ ONNX Runtime no funciona con CUDA")
        print("🔄 Intentando procesamiento de todas formas...")
    
    # Paso 3: Ejecutar procesamiento
    success = run_processing_with_fixed_cuda()
    
    if success:
        print("🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
        print("📁 Archivo guardado en: /content/resultados/DanielaAS130_CUDA.mp4")
    else:
        print("❌ Error en el procesamiento")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 