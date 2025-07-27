#!/usr/bin/env python3
"""
Configurar PyTorch GPU para ROOP en Google Colab
"""

import os
import sys
import subprocess

def install_pytorch_gpu():
    """Instalar PyTorch GPU para Colab"""
    print("🚀 INSTALANDO PYTORCH GPU PARA COLAB:")
    print("=" * 60)
    
    try:
        # Desinstalar TensorFlow para evitar conflictos
        print("⏳ Desinstalando TensorFlow...")
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar PyTorch GPU
        print("⏳ Instalando PyTorch GPU...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ PyTorch GPU instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando PyTorch: {e}")
        return False

def install_roop_dependencies():
    """Instalar dependencias necesarias para ROOP"""
    print("\n📦 INSTALANDO DEPENDENCIAS ROOP:")
    print("=" * 40)
    
    try:
        # Instalar dependencias necesarias
        dependencies = [
            "numpy>=1.25.2",
            "opencv-python",
            "pillow",
            "scikit-image",
            "scipy",
            "onnxruntime-gpu",
            "insightface",
            "opennsfw2",
            "nvidia-ml-py3",
            "pynvml"
        ]
        
        for dep in dependencies:
            print(f"⏳ Instalando {dep}...")
            cmd = [sys.executable, "-m", "pip", "install", dep]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Dependencias ROOP instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_pytorch_gpu_env():
    """Configurar entorno PyTorch GPU"""
    print("\n🔧 CONFIGURANDO ENTORNO PYTORCH GPU:")
    print("=" * 40)
    
    # Variables de entorno para PyTorch GPU
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDA_LAUNCH_BLOCKING': '1',
        'TORCH_CUDNN_V8_API_ENABLED': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno PyTorch configuradas")

def test_pytorch_gpu():
    """Probar PyTorch GPU"""
    print("\n🧪 PROBANDO PYTORCH GPU:")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"🎮 CUDA disponible: {torch.cuda.get_device_count()} dispositivos")
            for i in range(torch.cuda.get_device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memoria: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
            
            # Probar operación en GPU
            device = torch.device('cuda:0')
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
            c = torch.matmul(a, b)
            print(f"✅ Operación PyTorch GPU exitosa: {c}")
            
            # Verificar memoria GPU
            print(f"📊 Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
            print(f"📊 Memoria GPU reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
            
            return True
        else:
            print("❌ CUDA no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error probando PyTorch GPU: {e}")
        return False

def create_pytorch_roop_wrapper():
    """Crear wrapper ROOP con PyTorch GPU"""
    print("\n📝 CREANDO WRAPPER ROOP PYTORCH GPU:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper para ROOP con PyTorch GPU en Google Colab
"""

import os
import sys
import subprocess

# CONFIGURACIÓN PYTORCH GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def check_pytorch_gpu():
    try:
        import torch
        print(f"🎮 PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_count()} dispositivos")
            for i in range(torch.cuda.get_device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Configurar GPU
            torch.cuda.empty_cache()
            print("✅ GPU configurado para ROOP")
            return True
        else:
            print("❌ CUDA no disponible")
            return False
    except Exception as e:
        print(f"❌ Error verificando PyTorch GPU: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ROOP CON PYTORCH GPU")
    print("=" * 40)
    
    # Verificar PyTorch GPU
    if not check_pytorch_gpu():
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
        with open('run_roop_pytorch_gpu.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper PyTorch GPU creado: run_roop_pytorch_gpu.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def update_roop_script_pytorch():
    """Actualizar script principal para usar PyTorch GPU"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar comando para usar wrapper PyTorch
        old_cmd = "sys.executable, 'run_roop_colab_gpu_final.py'"
        new_cmd = "sys.executable, 'run_roop_pytorch_gpu.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
        else:
            # Buscar otros comandos
            old_cmds = [
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
        
        print("✅ Script actualizado para usar PyTorch GPU")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def create_gpu_monitor():
    """Crear script de monitoreo GPU"""
    print("\n📊 CREANDO MONITOR GPU:")
    print("=" * 40)
    
    monitor_content = '''#!/usr/bin/env python3
"""
Monitor GPU para ROOP
"""

import time
import psutil
import pynvml

def monitor_gpu():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        while True:
            # Información GPU
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Información RAM
            ram = psutil.virtual_memory()
            
            print(f"\\r🎮 GPU: {info.used/1024**3:.1f}GB/{info.total/1024**3:.1f}GB "
                  f"({utilization.gpu}% util) | "
                  f"💾 RAM: {ram.percent}%", end='', flush=True)
            
            time.sleep(2)
    except:
        pass

if __name__ == '__main__':
    monitor_gpu()
'''
    
    try:
        with open('monitor_gpu.py', 'w') as f:
            f.write(monitor_content)
        print("✅ Monitor GPU creado: monitor_gpu.py")
        return True
    except Exception as e:
        print(f"❌ Error creando monitor: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 CONFIGURANDO PYTORCH GPU PARA ROOP")
    print("=" * 60)
    
    # Instalar PyTorch GPU
    if not install_pytorch_gpu():
        print("❌ Error instalando PyTorch GPU")
        return False
    
    # Instalar dependencias ROOP
    if not install_roop_dependencies():
        print("❌ Error instalando dependencias ROOP")
        return False
    
    # Configurar entorno PyTorch GPU
    setup_pytorch_gpu_env()
    
    # Crear wrapper PyTorch GPU
    if not create_pytorch_roop_wrapper():
        print("❌ Error creando wrapper PyTorch")
        return False
    
    # Actualizar script principal
    update_roop_script_pytorch()
    
    # Crear monitor GPU
    create_gpu_monitor()
    
    # Probar PyTorch GPU
    if not test_pytorch_gpu():
        print("❌ Error: PyTorch GPU no funciona")
        return False
    
    print("\n✅ PYTORCH GPU CONFIGURADO EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("2. O usar wrapper directamente: python run_roop_pytorch_gpu.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Monitorear GPU: python monitor_gpu.py")
    print("4. Verificar uso GPU: Deberías ver 8-12 GB de VRAM en uso")
    
    return True

if __name__ == '__main__':
    main() 