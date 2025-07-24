#!/usr/bin/env python3
"""
Script para diagnosticar y forzar el uso de GPU en face-swapper
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_onnx_configuration():
    """Verificar configuración de ONNX Runtime"""
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN ONNX")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        
        # Verificar proveedores disponibles
        providers = ort.get_available_providers()
        print(f"✅ Proveedores disponibles: {providers}")
        
        # Verificar CUDA específicamente
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible")
            
            # Verificar opciones de CUDA
            try:
                cuda_options = ort.get_provider_options('CUDAExecutionProvider')
                print(f"📊 Opciones CUDA: {cuda_options}")
            except Exception as e:
                print(f"⚠️ No se pudieron obtener opciones CUDA: {e}")
        else:
            print("❌ CUDA GPU NO disponible")
            
        # Verificar versión de ONNX Runtime
        print(f"📦 Versión ONNX Runtime: {ort.__version__}")
        
        # Verificar si es la versión GPU
        if 'gpu' in ort.__version__.lower() or 'cuda' in ort.__version__.lower():
            print("✅ ONNX Runtime GPU detectado")
        else:
            print("⚠️ Posible ONNX Runtime CPU - considera instalar onnxruntime-gpu")
            
    except ImportError as e:
        print(f"❌ Error importando ONNX Runtime: {e}")
        print("💡 Instala: pip install onnxruntime-gpu")

def check_face_swapper_gpu_usage():
    """Verificar si face-swapper está usando GPU"""
    print("\n🔍 VERIFICANDO USO DE GPU EN FACE-SWAPPER")
    print("=" * 50)
    
    # Verificar archivos del face-swapper
    face_swapper_path = Path("roop/processors/frame/face_swapper.py")
    if face_swapper_path.exists():
        print(f"✅ Face-swapper encontrado: {face_swapper_path}")
        
        # Buscar configuración de GPU en el código
        with open(face_swapper_path, 'r') as f:
            content = f.read()
            
        if 'CUDAExecutionProvider' in content:
            print("✅ Face-swapper tiene configuración CUDA")
        else:
            print("⚠️ Face-swapper NO tiene configuración CUDA explícita")
            
        if 'onnxruntime' in content:
            print("✅ Face-swapper usa ONNX Runtime")
        else:
            print("❌ Face-swapper NO usa ONNX Runtime")
    else:
        print(f"❌ Face-swapper no encontrado: {face_swapper_path}")

def create_gpu_forced_command(source_path, target_path, output_path):
    """Crear comando que fuerza el uso de GPU"""
    
    print("\n🚀 COMANDO PARA FORZAR USO DE GPU")
    print("=" * 50)
    
    # Variables de entorno para forzar GPU
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'OMP_NUM_THREADS': '1',
        'ONNXRUNTIME_PROVIDER_SHARED_LIB': '/usr/local/cuda/lib64/libonnxruntime_providers_cuda.so'
    }
    
    # Comando optimizado para GPU
    cmd = [
        "roop_env/bin/python", 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper',  # Solo face-swapper primero
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '5',
        '--temp-frame-quality', '100',
        '--temp-frame-format', 'png',
        '--output-video-encoder', 'h264_nvenc',
        '--output-video-quality', '100',
        '--keep-fps'
    ]
    
    print("Variables de entorno:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    
    print("\nComando:")
    print(" ".join(cmd))
    
    return cmd, env_vars

def test_gpu_with_small_video():
    """Probar GPU con un video pequeño"""
    
    print("\n🧪 PRUEBA CON VIDEO PEQUEÑO")
    print("=" * 40)
    
    # Crear comando de prueba
    test_source = "/content/DanielaAS.jpg"
    test_target = "/content/112.mp4"
    test_output = "/content/test_gpu_output.mp4"
    
    cmd, env_vars = create_gpu_forced_command(test_source, test_target, test_output)
    
    print(f"\n📝 Comando de prueba:")
    print(" ".join(cmd))
    
    print(f"\n💡 Para probar:")
    print("1. Ejecuta el comando anterior")
    print("2. Monitorea con: nvidia-smi -l 1")
    print("3. Verifica que VRAM > 0GB durante el procesamiento")
    print("4. Si VRAM sigue en 0GB, hay un problema de configuración")

def check_dependencies():
    """Verificar dependencias necesarias"""
    
    print("\n📦 VERIFICANDO DEPENDENCIAS")
    print("=" * 40)
    
    # Verificar CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit instalado")
        else:
            print("❌ CUDA Toolkit NO instalado")
    except FileNotFoundError:
        print("❌ CUDA Toolkit NO encontrado")
    
    # Verificar nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi disponible")
        else:
            print("❌ nvidia-smi NO disponible")
    except FileNotFoundError:
        print("❌ nvidia-smi NO encontrado")
    
    # Verificar onnxruntime-gpu
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime instalado: {ort.__version__}")
    except ImportError:
        print("❌ ONNX Runtime NO instalado")

def main():
    print("🔧 DIAGNÓSTICO Y FORZADO DE GPU PARA FACE-SWAPPER")
    print("=" * 60)
    
    # Verificar configuración
    check_onnx_configuration()
    check_face_swapper_gpu_usage()
    check_dependencies()
    
    # Crear comando optimizado
    cmd, env_vars = create_gpu_forced_command(
        "/content/DanielaAS.jpg",
        "/content/112.mp4",
        "/content/DanielaAS112_gpu.mp4"
    )
    
    print("\n" + "=" * 60)
    print("💡 DIAGNÓSTICO:")
    print("=" * 60)
    print("❌ PROBLEMA: Face-swapper no está usando GPU")
    print("📊 SÍNTOMAS:")
    print("  • VRAM = 0.0GB durante procesamiento")
    print("  • Velocidad lenta (6.3s/frame)")
    print("  • Proveedor CUDA disponible pero no usado")
    print("\n🔧 SOLUCIONES:")
    print("1. Verificar que onnxruntime-gpu esté instalado")
    print("2. Usar el comando optimizado con variables de entorno")
    print("3. Monitorear con nvidia-smi durante el procesamiento")
    print("4. Si persiste, puede ser un problema del modelo face-swapper")
    
    print("\n" + "=" * 60)
    print("🚀 PRÓXIMOS PASOS:")
    print("=" * 60)
    print("1. Ejecuta: python fix_face_swapper_gpu.py")
    print("2. Usa el comando optimizado generado")
    print("3. Monitorea con: nvidia-smi -l 1")
    print("4. Verifica que VRAM > 0GB durante el procesamiento")

if __name__ == "__main__":
    main() 