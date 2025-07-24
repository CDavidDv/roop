#!/usr/bin/env python3
"""
Script para corregir la instalación de ONNX Runtime GPU
"""

import subprocess
import sys
import os

def print_status(message, status="INFO"):
    """Imprimir mensaje de estado"""
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{emoji.get(status, 'ℹ️')} {message}")

def check_current_onnx():
    """Verificar ONNX Runtime actual"""
    print_status("Verificando ONNX Runtime actual...", "INFO")
    
    try:
        result = subprocess.run([
            'roop_env/bin/python', '-c', 
            "import onnxruntime as ort; print('Versión:', ort.__version__); print('GPU:', 'gpu' in ort.__version__.lower()); print('Proveedores:', ort.get_available_providers())"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            if 'GPU: False' in result.stdout:
                print_status("❌ ONNX Runtime CPU detectado", "ERROR")
                return False
            elif 'GPU: True' in result.stdout:
                print_status("✅ ONNX Runtime GPU detectado", "SUCCESS")
                return True
            else:
                print_status("⚠️ No se pudo determinar la versión", "WARNING")
                return False
        else:
            print_status(f"Error verificando ONNX: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        return False

def uninstall_onnx_cpu():
    """Desinstalar ONNX Runtime CPU"""
    print_status("Desinstalando ONNX Runtime CPU...", "INFO")
    
    # Desinstalar todas las versiones
    subprocess.run(['roop_env/bin/pip', 'uninstall', 'onnxruntime', '-y'])
    subprocess.run(['roop_env/bin/pip', 'uninstall', 'onnxruntime-gpu', '-y'])
    subprocess.run(['roop_env/bin/pip', 'uninstall', 'onnxruntime-cpu', '-y'])
    
    print_status("ONNX Runtime desinstalado", "SUCCESS")

def install_onnx_gpu():
    """Instalar ONNX Runtime GPU"""
    print_status("Instalando ONNX Runtime GPU...", "INFO")
    
    # Instalar versión específica de GPU
    result = subprocess.run([
        'roop_env/bin/pip', 'install', 'onnxruntime-gpu==1.15.1', '--no-cache-dir', '--force-reinstall'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print_status("ONNX Runtime GPU instalado", "SUCCESS")
        return True
    else:
        print_status(f"Error instalando: {result.stderr}", "ERROR")
        return False

def verify_gpu_installation():
    """Verificar instalación GPU"""
    print_status("Verificando instalación GPU...", "INFO")
    
    try:
        result = subprocess.run([
            'roop_env/bin/python', '-c', 
            "import onnxruntime as ort; print('Versión:', ort.__version__); print('GPU:', 'gpu' in ort.__version__.lower()); print('Proveedores:', ort.get_available_providers())"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            if 'GPU: True' in result.stdout:
                print_status("✅ ONNX Runtime GPU instalado correctamente", "SUCCESS")
                return True
            else:
                print_status("❌ ONNX Runtime sigue siendo CPU", "ERROR")
                return False
        else:
            print_status(f"Error verificando: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        return False

def test_gpu_functionality():
    """Probar funcionalidad GPU"""
    print_status("Probando funcionalidad GPU...", "INFO")
    
    test_script = '''
import onnxruntime as ort
import numpy as np

print("🔍 Probando GPU...")

# Verificar proveedores
providers = ort.get_available_providers()
print(f"Proveedores disponibles: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ CUDA disponible")
    
    # Probar sesión simple
    try:
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        session_options = ort.SessionOptions()
        
        # Crear modelo simple
        import onnx
        from onnx import helper
        
        X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph)
        
        # Probar con CUDA
        session = ort.InferenceSession(
            model.SerializeToString(),
            session_options,
            providers=['CUDAExecutionProvider']
        )
        
        print("✅ Sesión CUDA creada exitosamente")
        print("✅ GPU funciona correctamente")
        
    except Exception as e:
        print(f"❌ Error probando sesión CUDA: {e}")
        
else:
    print("❌ CUDA NO disponible")
'''
    
    result = subprocess.run(['roop_env/bin/python', '-c', test_script], capture_output=True, text=True)
    print(result.stdout)
    
    if '✅ GPU funciona correctamente' in result.stdout:
        return True
    else:
        return False

def create_high_quality_command():
    """Crear comando con alta calidad"""
    print_status("Creando comando con alta calidad...", "INFO")
    
    cmd = '''roop_env/bin/python run.py \\
  --source /content/DanielaAS.jpg \\
  --target /content/112.mp4 \\
  -o /content/DanielaAS112_high_quality.mp4 \\
  --frame-processor face_swapper \\
  --execution-provider cuda \\
  --max-memory 8 \\
  --execution-threads 8 \\
  --gpu-memory-wait 15 \\
  --temp-frame-format png \\
  --temp-frame-quality 0 \\
  --output-video-encoder libx264 \\
  --output-video-quality 35 \\
  --keep-fps'''
    
    with open('run_high_quality_gpu.sh', 'w') as f:
        f.write(cmd)
    
    os.chmod('run_high_quality_gpu.sh', 0o755)
    print_status("Comando guardado en: run_high_quality_gpu.sh", "SUCCESS")
    
    print("\n🎬 COMANDO CON ALTA CALIDAD:")
    print("=" * 50)
    print(cmd)

def main():
    """Función principal"""
    print("🔧 CORRECCIÓN DE INSTALACIÓN ONNX RUNTIME GPU")
    print("=" * 60)
    
    # Verificar instalación actual
    if check_current_onnx():
        print_status("ONNX Runtime GPU ya está instalado correctamente", "SUCCESS")
        return
    
    # Desinstalar CPU
    uninstall_onnx_cpu()
    
    # Instalar GPU
    if not install_onnx_gpu():
        print_status("Error instalando ONNX Runtime GPU", "ERROR")
        return
    
    # Verificar instalación
    if not verify_gpu_installation():
        print_status("Error verificando instalación GPU", "ERROR")
        return
    
    # Probar funcionalidad
    if not test_gpu_functionality():
        print_status("Error probando funcionalidad GPU", "ERROR")
        return
    
    # Crear comando optimizado
    create_high_quality_command()
    
    print_status("¡INSTALACIÓN CORREGIDA EXITOSAMENTE!", "SUCCESS")
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. Ejecutar: ./run_high_quality_gpu.sh")
    print("2. Monitorear: nvidia-smi -l 1")
    print("3. Verificar que VRAM > 0GB durante procesamiento")

if __name__ == "__main__":
    main() 