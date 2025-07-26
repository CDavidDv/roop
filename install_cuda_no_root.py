#!/usr/bin/env python3
"""
Instalar librerías CUDA sin root
"""

import subprocess
import sys
import os

def install_cuda_no_root():
    """Instala librerías CUDA sin root"""
    print("🔧 INSTALANDO LIBRERÍAS CUDA SIN ROOT")
    print("=" * 50)
    
    try:
        # 1. Reinstalar onnxruntime-gpu
        print("1. Reinstalando onnxruntime-gpu...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.16.3"
        ], check=True)
        print("✅ onnxruntime-gpu reinstalado")
        
        # 2. Instalar librerías CUDA con conda si está disponible
        print("2. Instalando librerías CUDA...")
        try:
            subprocess.run([
                "conda", "install", "-y", "-c", "conda-forge", "cudatoolkit=11.8", "cudnn=8.9"
            ], check=True)
            print("✅ Librerías CUDA instaladas con conda")
        except:
            print("⚠️ Conda no disponible, intentando con pip...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12==8.9.4.25"
                ], check=True)
                print("✅ Librerías CUDA instaladas con pip")
            except:
                print("⚠️ No se pudieron instalar librerías CUDA")
        
        # 3. Crear enlaces simbólicos en directorio local
        print("3. Creando enlaces simbólicos...")
        local_lib_dir = os.path.expanduser("~/cuda_libs")
        os.makedirs(local_lib_dir, exist_ok=True)
        
        # Buscar librerías CUDA en el sistema
        cuda_libs = [
            "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11",
            "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
            "/usr/lib/x86_64-linux-gnu/libcufft.so.10",
            "/usr/local/cuda-11.8/lib64/libcublasLt.so.11",
            "/usr/local/cuda-11.8/lib64/libcudnn.so.8",
            "/usr/local/cuda-11.8/lib64/libcufft.so.10",
        ]
        
        for lib in cuda_libs:
            if os.path.exists(lib):
                target = os.path.join(local_lib_dir, os.path.basename(lib))
                try:
                    os.symlink(lib, target)
                    print(f"✅ Enlace creado: {target}")
                except:
                    pass
        
        # 4. Configurar variables de entorno
        print("4. Configurando variables de entorno...")
        os.environ['LD_LIBRARY_PATH'] = f"{local_lib_dir}:" + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        
        print("✅ Variables de entorno configuradas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cuda_installation():
    """Prueba la instalación de CUDA"""
    print("\n🧪 PROBANDO INSTALACIÓN CUDA")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        
        # Verificar proveedores
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores disponibles: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ CUDA disponible")
            
            # Probar creación de sesión CUDA
            import numpy as np
            test_data = np.random.rand(1, 3, 128, 128).astype(np.float32)
            
            # Crear sesión simple para probar CUDA
            session = ort.InferenceSession(
                "models/inswapper_128.onnx",
                providers=['CUDAExecutionProvider']
            )
            print("✅ Sesión CUDA creada exitosamente")
            
            return True
        else:
            print("❌ CUDA no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error probando CUDA: {e}")
        return False

def main():
    """Función principal"""
    if install_cuda_no_root():
        if test_cuda_installation():
            print("\n🎉 ¡CUDA INSTALADO Y FUNCIONANDO!")
            print("=" * 40)
            print("✅ Librerías CUDA instaladas")
            print("✅ GPU disponible")
            print("✅ Listo para procesar")
            print("\n🚀 Ejecuta:")
            print("python force_gpu.py")
            return 0
        else:
            print("\n❌ CUDA no funciona correctamente")
            return 1
    else:
        print("\n❌ Error instalando CUDA")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 