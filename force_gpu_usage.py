#!/usr/bin/env python3
"""
Script para forzar el uso de GPU en ROOP
"""

import os
import sys
import subprocess

def force_gpu_settings():
    """Configurar variables de entorno para forzar uso de GPU"""
    
    # Variables de entorno para forzar GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print("🔧 CONFIGURANDO FORZADO DE GPU")
    print("=" * 50)
    print("✅ CUDA_VISIBLE_DEVICES = 0")
    print("✅ TF_FORCE_GPU_ALLOW_GROWTH = true")
    print("✅ TF_CPP_MIN_LOG_LEVEL = 2")
    print("✅ OMP_NUM_THREADS = 1")

def check_onnx_providers():
    """Verificar proveedores de ONNX Runtime"""
    print("\n🔍 VERIFICANDO PROVEEDORES ONNX:")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible")
        else:
            print("❌ CUDA GPU no disponible")
            
        # Verificar configuración de CUDA
        cuda_provider_options = ort.get_provider_options('CUDAExecutionProvider')
        print(f"Opciones CUDA: {cuda_provider_options}")
        
    except Exception as e:
        print(f"❌ Error verificando ONNX: {e}")

def create_optimized_command(source_path, target_path, output_path):
    """Crear comando optimizado para forzar uso de GPU"""
    
    # Configurar variables de entorno
    force_gpu_settings()
    
    # Comando optimizado para GPU
    cmd = [
        "roop_env/bin/python", 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--execution-provider', 'cuda',  # Forzar CUDA
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '5',
        '--temp-frame-quality', '100',
        '--temp-frame-format', 'png',
        '--output-video-encoder', 'h264_nvenc',
        '--output-video-quality', '100',
        '--keep-fps'
    ]
    
    return cmd

def test_gpu_usage():
    """Probar uso de GPU con un video pequeño"""
    
    print("\n🧪 PRUEBA DE USO DE GPU")
    print("=" * 40)
    
    # Verificar configuración
    check_onnx_providers()
    
    print("\n💡 CONSEJOS PARA FORZAR GPU:")
    print("1. Asegúrate de que CUDA esté instalado correctamente")
    print("2. Verifica que onnxruntime-gpu esté instalado")
    print("3. Usa --execution-provider cuda explícitamente")
    print("4. Monitorea con nvidia-smi durante el procesamiento")
    
    # Comando de prueba
    test_cmd = [
        "python", "-c",
        "import onnxruntime as ort; print('CUDA disponible:', 'CUDAExecutionProvider' in ort.get_available_providers())"
    ]
    
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        print(f"\nResultado de prueba: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error en prueba: {e}")

def main():
    print("🔧 FORZADOR DE USO DE GPU PARA ROOP")
    print("=" * 50)
    
    # Verificar configuración
    check_onnx_providers()
    test_gpu_usage()
    
    print("\n" + "=" * 50)
    print("🚀 COMANDO OPTIMIZADO PARA GPU:")
    print("=" * 50)
    
    # Ejemplo de comando
    example_cmd = create_optimized_command(
        "/content/LilitAS.png",
        "/content/62.mp4", 
        "/content/resultados/LilitAS62.mp4"
    )
    
    print("Comando recomendado:")
    print(" ".join(example_cmd))
    
    print("\n" + "=" * 50)
    print("📊 MONITOREO DURANTE PROCESAMIENTO:")
    print("=" * 50)
    print("1. Ejecuta en una terminal:")
    print("   nvidia-smi -l 1")
    print("\n2. Ejecuta en otra terminal:")
    print("   python check_gpu_usage.py")
    print("\n3. Verifica que VRAM > 0GB durante el procesamiento")

if __name__ == "__main__":
    main() 