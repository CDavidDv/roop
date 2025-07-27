#!/usr/bin/env python3
"""
Configuración completa de GPU para ROOP
"""

import os
import sys
import subprocess

def check_gpu_requirements():
    """Verificar requisitos GPU"""
    print("🔍 VERIFICANDO REQUISITOS GPU:")
    print("=" * 40)
    
    # Verificar nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi disponible")
            print(result.stdout)
        else:
            print("❌ nvidia-smi no disponible")
            return False
    except:
        print("❌ nvidia-smi no encontrado")
        return False
    
    # Verificar CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA disponible")
            print(result.stdout)
        else:
            print("❌ CUDA no disponible")
            return False
    except:
        print("❌ CUDA no encontrado")
        return False
    
    return True

def install_complete_gpu_stack():
    """Instalar stack completo de GPU"""
    print("\n📦 INSTALANDO STACK COMPLETO GPU:")
    print("=" * 40)
    
    try:
        # Desinstalar todo lo problemático
        print("⏳ Limpiando instalaciones anteriores...")
        packages_to_remove = [
            "tensorflow", "tensorflow-gpu", "torch", "torchvision", "torchaudio",
            "numpy", "onnxruntime", "onnxruntime-gpu"
        ]
        
        for package in packages_to_remove:
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Instalar NumPy compatible
        print("⏳ Instalando NumPy 1.24.3...")
        cmd1 = [sys.executable, "-m", "pip", "install", "numpy==1.24.3"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar PyTorch GPU
        print("⏳ Instalando PyTorch 2.0.1 GPU...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        # Instalar ONNX Runtime GPU
        print("⏳ Instalando ONNX Runtime GPU...")
        cmd3 = [sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.15.1"]
        subprocess.run(cmd3, check=True, capture_output=True, text=True)
        
        # Instalar dependencias ROOP
        print("⏳ Instalando dependencias ROOP...")
        roop_deps = [
            "opencv-python",
            "pillow",
            "scikit-image",
            "scipy",
            "insightface",
            "opennsfw2",
            "nvidia-ml-py3",
            "pynvml"
        ]
        
        for dep in roop_deps:
            cmd = [sys.executable, "-m", "pip", "install", dep]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Stack GPU completo instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando stack GPU: {e}")
        return False

def configure_gpu_environment():
    """Configurar entorno GPU completo"""
    print("\n🔧 CONFIGURANDO ENTORNO GPU COMPLETO:")
    print("=" * 40)
    
    # Variables de entorno para GPU completo
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_CUDNN_V8_API_ENABLED': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        # Variables para ONNX Runtime
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        # Variables para InsightFace
        'INSIGHTFACE_HOME': '/root/.insightface',
        # Variables para OpenCV
        'OPENCV_OPENCL_RUNTIME': '',
        'OPENCV_OPENCL_DEVICE': ':GPU:0'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno GPU configuradas")

def test_complete_gpu_stack():
    """Probar stack GPU completo"""
    print("\n🧪 PROBANDO STACK GPU COMPLETO:")
    print("=" * 40)
    
    try:
        # Probar NumPy
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        # Probar PyTorch GPU
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🎮 CUDA disponible: {torch.cuda.device_count()} dispositivos")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memoria: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
            
            # Probar operación PyTorch GPU
            device = torch.device('cuda:0')
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
            c = torch.matmul(a, b)
            print(f"✅ Operación PyTorch GPU exitosa: {c}")
        else:
            print("❌ CUDA no disponible para PyTorch")
            return False
        
        # Probar ONNX Runtime GPU
        import onnxruntime as ort
        print(f"✅ ONNX Runtime version: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"📱 Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU disponible")
        else:
            print("❌ ONNX Runtime GPU no disponible")
            return False
        
        # Probar InsightFace
        try:
            import insightface
            print("✅ InsightFace disponible")
        except Exception as e:
            print(f"❌ Error con InsightFace: {e}")
            return False
        
        print("✅ Stack GPU completo funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Error probando stack GPU: {e}")
        return False

def create_roop_gpu_wrapper():
    """Crear wrapper ROOP con GPU completo"""
    print("\n📝 CREANDO WRAPPER ROOP GPU COMPLETO:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper completo para ROOP con GPU
"""

import os
import sys
import subprocess

# CONFIGURACIÓN GPU COMPLETA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['INSIGHTFACE_HOME'] = '/root/.insightface'
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
os.environ['OPENCV_OPENCL_DEVICE'] = ':GPU:0'

def check_gpu_stack():
    try:
        import torch
        import onnxruntime as ort
        
        print(f"🎮 PyTorch version: {torch.__version__}")
        print(f"📱 ONNX Runtime version: {ort.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.device_count()} dispositivos")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU disponible")
        else:
            print("❌ ONNX Runtime GPU no disponible")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ROOP CON GPU COMPLETO")
    print("=" * 40)
    
    # Verificar GPU stack
    if not check_gpu_stack():
        print("❌ No se puede continuar sin GPU")
        return False
    
    # Obtener argumentos
    args = sys.argv[1:]
    
    # Construir comando
    cmd = [sys.executable, 'run.py'] + args
    
    print(f"🚀 Ejecutando: {' '.join(cmd)}")
    print("=" * 40)
    
    try:
        # Ejecutar ROOP
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ ROOP ejecutado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando ROOP:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

if __name__ == '__main__':
    main()
'''
    
    try:
        with open('run_roop_gpu_complete.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper GPU completo creado: run_roop_gpu_complete.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def update_roop_script_complete():
    """Actualizar script principal para usar wrapper completo"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar comando para usar wrapper completo
        old_cmd = "sys.executable, 'run_roop_simple_gpu.py'"
        new_cmd = "sys.executable, 'run_roop_gpu_complete.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
        else:
            # Buscar otros comandos
            old_cmds = [
                "sys.executable, 'run_roop_pytorch_gpu.py'",
                "sys.executable, 'run_roop_colab_gpu_final.py'",
                "sys.executable, 'run_roop_colab_gpu.py'",
                "sys.executable, 'run_roop_gpu_forced.py'",
                "sys.executable, 'run_roop_wrapper.py'"
            ]
            for old_cmd in old_cmds:
                if old_cmd in content:
                    content = content.replace(old_cmd, new_cmd)
                    break
            else:
                print("⚠️ No se encontró comando a reemplazar")
                return False
        
        with open(script_file, 'w') as f:
            f.write(content)
        
        print("✅ Script actualizado para usar wrapper completo")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def create_gpu_test_complete():
    """Crear prueba GPU completa"""
    print("\n🧪 CREANDO PRUEBA GPU COMPLETA:")
    print("=" * 40)
    
    test_content = '''#!/usr/bin/env python3
"""
Prueba completa de GPU para ROOP
"""

import os
import torch
import onnxruntime as ort

def test_gpu_complete():
    print("🧪 PROBANDO GPU COMPLETO")
    print("=" * 40)
    
    try:
        # Probar PyTorch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🎮 CUDA disponible: {torch.cuda.device_count()} dispositivos")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memoria: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
            
            # Probar operación PyTorch
            device = torch.device('cuda:0')
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
            c = torch.matmul(a, b)
            print(f"✅ Operación PyTorch GPU exitosa: {c}")
        else:
            print("❌ CUDA no disponible para PyTorch")
            return False
        
        # Probar ONNX Runtime
        print(f"✅ ONNX Runtime version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"📱 Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU disponible")
        else:
            print("❌ ONNX Runtime GPU no disponible")
            return False
        
        print("✅ GPU completo funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    test_gpu_complete()
'''
    
    try:
        with open('test_gpu_complete.py', 'w') as f:
            f.write(test_content)
        print("✅ Prueba GPU completa creada: test_gpu_complete.py")
        return True
    except Exception as e:
        print(f"❌ Error creando prueba: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 CONFIGURACIÓN COMPLETA GPU PARA ROOP")
    print("=" * 60)
    
    # Verificar requisitos GPU
    if not check_gpu_requirements():
        print("❌ Requisitos GPU no cumplidos")
        return False
    
    # Instalar stack GPU completo
    if not install_complete_gpu_stack():
        print("❌ Error instalando stack GPU")
        return False
    
    # Configurar entorno GPU
    configure_gpu_environment()
    
    # Crear wrapper completo
    if not create_roop_gpu_wrapper():
        print("❌ Error creando wrapper completo")
        return False
    
    # Actualizar script principal
    update_roop_script_complete()
    
    # Crear prueba GPU completa
    create_gpu_test_complete()
    
    # Probar stack GPU completo
    if not test_complete_gpu_stack():
        print("❌ Error: Stack GPU no funciona")
        return False
    
    print("\n✅ CONFIGURACIÓN GPU COMPLETA EXITOSA")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Probar GPU: python test_gpu_complete.py")
    print("2. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("3. O usar wrapper directamente: python run_roop_gpu_complete.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("4. Verificar uso GPU: Deberías ver 8-12 GB de VRAM en uso")
    
    return True

if __name__ == '__main__':
    main() 