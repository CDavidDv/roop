#!/usr/bin/env python3
"""
Arreglar conflicto NumPy y PyTorch para ROOP
"""

import os
import sys
import subprocess

def fix_numpy_conflict():
    """Arreglar conflicto de NumPy"""
    print("🔧 ARREGLANDO CONFLICTO NUMPY:")
    print("=" * 40)
    
    try:
        # Desinstalar NumPy actual
        print("⏳ Desinstalando NumPy actual...")
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "numpy"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar NumPy compatible
        print("⏳ Instalando NumPy 1.24.3...")
        cmd2 = [sys.executable, "-m", "pip", "install", "numpy==1.24.3"]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ NumPy arreglado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error arreglando NumPy: {e}")
        return False

def reinstall_pytorch_compatible():
    """Reinstalar PyTorch compatible"""
    print("\n📦 REINSTALANDO PYTORCH COMPATIBLE:")
    print("=" * 40)
    
    try:
        # Desinstalar PyTorch actual
        print("⏳ Desinstalando PyTorch actual...")
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar PyTorch compatible
        print("⏳ Instalando PyTorch 2.0.1...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ PyTorch compatible instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando PyTorch: {e}")
        return False

def test_pytorch_fixed():
    """Probar PyTorch arreglado"""
    print("\n🧪 PROBANDO PYTORCH ARREGLADO:")
    print("=" * 40)
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"🎮 CUDA disponible: {torch.cuda.device_count()} dispositivos")
            for i in range(torch.cuda.device_count()):
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
        print(f"❌ Error probando PyTorch: {e}")
        return False

def create_simple_roop_wrapper():
    """Crear wrapper simple para ROOP"""
    print("\n📝 CREANDO WRAPPER SIMPLE ROOP:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper simple para ROOP con GPU
"""

import os
import sys
import subprocess

# CONFIGURACIÓN SIMPLE GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    """Función principal"""
    print("🚀 ROOP CON GPU SIMPLE")
    print("=" * 40)
    
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
        with open('run_roop_simple_gpu.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper simple creado: run_roop_simple_gpu.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def update_roop_script_simple():
    """Actualizar script principal para usar wrapper simple"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar comando para usar wrapper simple
        old_cmd = "sys.executable, 'run_roop_pytorch_gpu.py'"
        new_cmd = "sys.executable, 'run_roop_simple_gpu.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
        else:
            # Buscar otros comandos
            old_cmds = [
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
        
        print("✅ Script actualizado para usar wrapper simple")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def create_gpu_test():
    """Crear script de prueba GPU simple"""
    print("\n🧪 CREANDO PRUEBA GPU SIMPLE:")
    print("=" * 40)
    
    test_content = '''#!/usr/bin/env python3
"""
Prueba simple de GPU
"""

import os
import torch

def test_gpu():
    print("🧪 PROBANDO GPU SIMPLE")
    print("=" * 40)
    
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🎮 CUDA disponible: {torch.cuda.device_count()} dispositivos")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memoria: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
            
            # Probar operación
            device = torch.device('cuda:0')
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            b = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=device)
            c = torch.matmul(a, b)
            print(f"✅ Operación GPU exitosa: {c}")
            
            return True
        else:
            print("❌ CUDA no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    test_gpu()
'''
    
    try:
        with open('test_gpu_simple.py', 'w') as f:
            f.write(test_content)
        print("✅ Prueba GPU simple creada: test_gpu_simple.py")
        return True
    except Exception as e:
        print(f"❌ Error creando prueba: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO NUMPY Y PYTORCH")
    print("=" * 60)
    
    # Arreglar conflicto NumPy
    if not fix_numpy_conflict():
        print("❌ Error arreglando NumPy")
        return False
    
    # Reinstalar PyTorch compatible
    if not reinstall_pytorch_compatible():
        print("❌ Error instalando PyTorch")
        return False
    
    # Crear wrapper simple
    if not create_simple_roop_wrapper():
        print("❌ Error creando wrapper simple")
        return False
    
    # Actualizar script principal
    update_roop_script_simple()
    
    # Crear prueba GPU
    create_gpu_test()
    
    # Probar PyTorch arreglado
    if not test_pytorch_fixed():
        print("❌ Error: PyTorch no funciona")
        return False
    
    print("\n✅ NUMPY Y PYTORCH ARREGLADOS EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Probar GPU: python test_gpu_simple.py")
    print("2. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("3. O usar wrapper directamente: python run_roop_simple_gpu.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    
    return True

if __name__ == '__main__':
    main() 