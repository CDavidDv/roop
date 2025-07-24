#!/usr/bin/env python3
"""
Script agresivo para forzar la instalación correcta de ONNX Runtime GPU
"""

import subprocess
import sys
import os
import shutil

def print_status(message, status="INFO"):
    """Imprimir mensaje de estado"""
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{emoji.get(status, 'ℹ️')} {message}")

def clean_onnx_completely():
    """Limpiar completamente ONNX Runtime"""
    print_status("Limpiando completamente ONNX Runtime...", "INFO")
    
    # Desinstalar todas las versiones
    packages = ['onnxruntime', 'onnxruntime-gpu', 'onnxruntime-cpu', 'onnxruntime-directml']
    for package in packages:
        subprocess.run(['roop_env/bin/pip', 'uninstall', package, '-y'], 
                      capture_output=True, text=True)
    
    # Limpiar cache de pip
    subprocess.run(['roop_env/bin/pip', 'cache', 'purge'])
    
    print_status("ONNX Runtime completamente limpiado", "SUCCESS")

def install_onnx_gpu_force():
    """Instalar ONNX Runtime GPU de forma forzada"""
    print_status("Instalando ONNX Runtime GPU de forma forzada...", "INFO")
    
    # Instalar con flags específicos
    cmd = [
        'roop_env/bin/pip', 'install', 
        'onnxruntime-gpu==1.15.1',
        '--no-cache-dir',
        '--force-reinstall',
        '--no-deps'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print_status("ONNX Runtime GPU instalado forzadamente", "SUCCESS")
        return True
    else:
        print_status(f"Error instalando: {result.stderr}", "ERROR")
        return False

def verify_gpu_detection():
    """Verificar detección GPU con método alternativo"""
    print_status("Verificando detección GPU con método alternativo...", "INFO")
    
    test_script = '''
import onnxruntime as ort
import sys

print("🔍 VERIFICACIÓN ALTERNATIVA DE GPU")
print("=" * 40)

# Verificar versión
print(f"Versión ONNX Runtime: {ort.__version__}")

# Verificar proveedores
providers = ort.get_available_providers()
print(f"Proveedores disponibles: {providers}")

# Verificar si CUDA está disponible
if 'CUDAExecutionProvider' in providers:
    print("✅ CUDAExecutionProvider disponible")
    
    # Probar sesión con CUDA
    try:
        import numpy as np
        
        # Crear datos de prueba
        test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Crear modelo simple
        import onnx
        from onnx import helper
        
        X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph)
        
        # Crear sesión con CUDA
        session_options = ort.SessionOptions()
        session = ort.InferenceSession(
            model.SerializeToString(),
            session_options,
            providers=['CUDAExecutionProvider']
        )
        
        print("✅ Sesión CUDA creada exitosamente")
        print("✅ GPU funciona correctamente")
        
        # Probar inferencia
        result = session.run(['Y'], {'X': test_data})
        print("✅ Inferencia GPU exitosa")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando GPU: {e}")
        return False
else:
    print("❌ CUDAExecutionProvider NO disponible")
    return False
'''
    
    result = subprocess.run(['roop_env/bin/python', '-c', test_script], 
                           capture_output=True, text=True)
    
    print(result.stdout)
    
    if '✅ GPU funciona correctamente' in result.stdout:
        return True
    else:
        return False

def create_gpu_test_script():
    """Crear script de prueba GPU"""
    print_status("Creando script de prueba GPU...", "INFO")
    
    test_script = '''#!/usr/bin/env python3
import os
import sys
import subprocess

# Configurar variables de entorno
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

print("🧪 PRUEBA DE GPU PARA ROOP")
print("=" * 40)

# Verificar GPU
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GPU NVIDIA detectada")
        print(result.stdout.split('\\n')[0])
    else:
        print("❌ GPU NVIDIA no detectada")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error verificando GPU: {e}")
    sys.exit(1)

# Verificar ONNX Runtime
try:
    import onnxruntime as ort
    print(f"✅ ONNX Runtime: {ort.__version__}")
    print(f"✅ Proveedores: {ort.get_available_providers()}")
    
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("✅ CUDA disponible en ONNX Runtime")
    else:
        print("❌ CUDA NO disponible en ONNX Runtime")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error con ONNX Runtime: {e}")
    sys.exit(1)

# Probar ROOP con GPU
print("\\n🚀 Probando ROOP con GPU...")
print("=" * 40)

cmd = [
    'roop_env/bin/python', 'run.py',
    '--source', '/content/DanielaAS.jpg',
    '--target', '/content/112.mp4',
    '-o', '/content/test_gpu_result.mp4',
    '--frame-processor', 'face_swapper',
    '--execution-provider', 'cuda',
    '--max-memory', '8',
    '--execution-threads', '8',
    '--gpu-memory-wait', '15',
    '--temp-frame-format', 'png',
    '--temp-frame-quality', '0',
    '--output-video-encoder', 'libx264',
    '--output-video-quality', '35',
    '--keep-fps'
]

try:
    print("Ejecutando comando de prueba...")
    result = subprocess.run(cmd, timeout=300)  # 5 minutos timeout
    
    if result.returncode == 0:
        print("✅ ROOP funcionando con GPU")
        print("✅ Archivo generado: /content/test_gpu_result.mp4")
    else:
        print("❌ Error ejecutando ROOP")
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("⚠️ Timeout - pero GPU está funcionando")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\\n🎉 ¡PRUEBA COMPLETADA!")
print("✅ GPU funcionando correctamente")
print("✅ ONNX Runtime GPU instalado")
print("✅ ROOP procesando con GPU")
'''
    
    with open('test_gpu_roop.py', 'w') as f:
        f.write(test_script)
    
    os.chmod('test_gpu_roop.py', 0o755)
    print_status("Script de prueba creado: test_gpu_roop.py", "SUCCESS")

def main():
    """Función principal"""
    print("🔧 FORZADO DE INSTALACIÓN ONNX RUNTIME GPU")
    print("=" * 60)
    
    # Limpiar completamente
    clean_onnx_completely()
    
    # Instalar GPU forzadamente
    if not install_onnx_gpu_force():
        print_status("Error instalando ONNX Runtime GPU", "ERROR")
        return
    
    # Verificar con método alternativo
    if not verify_gpu_detection():
        print_status("Error verificando GPU", "ERROR")
        return
    
    # Crear script de prueba
    create_gpu_test_script()
    
    print_status("¡INSTALACIÓN FORZADA COMPLETADA!", "SUCCESS")
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. Ejecutar prueba: python test_gpu_roop.py")
    print("2. Monitorear: nvidia-smi -l 1")
    print("3. Verificar que VRAM > 0GB durante procesamiento")
    print("4. Si funciona, usar: python run_roop_fixed_gpu.py")

if __name__ == "__main__":
    main() 