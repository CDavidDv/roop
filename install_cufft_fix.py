#!/usr/bin/env python3
"""
Script para instalar libcufft y habilitar GPU
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """Ejecuta un comando y maneja errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Exitoso")
            return True
        else:
            print(f"❌ {description} - Error")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Excepción: {e}")
        return False

def install_cufft():
    """Instala libcufft específicamente"""
    print("🔧 INSTALANDO LIBCUFFT PARA HABILITAR GPU")
    print("=" * 50)
    
    # Actualizar repositorios
    if not run_command("apt-get update", "Actualizando repositorios"):
        return False
    
    # Instalar libcufft específicamente
    cufft_packages = [
        "libcufft-11-8",
        "libcufft-dev-11-8",
        "libcurand-11-8",
        "libcurand-dev-11-8",
    ]
    
    for package in cufft_packages:
        if not run_command(f"apt-get install -y {package}", f"Instalando {package}"):
            print(f"⚠️ {package} no disponible, continuando...")
    
    return True

def verify_gpu_after_fix():
    """Verifica que GPU funcione después del fix"""
    print("🔍 VERIFICANDO GPU DESPUÉS DEL FIX")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider disponible - GPU habilitado!")
            return True
        else:
            print("❌ CUDA provider no disponible")
            return False
        
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def create_restart_script():
    """Crea un script para reiniciar el procesamiento con GPU"""
    print("🔄 CREANDO SCRIPT DE REINICIO CON GPU")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script para reiniciar procesamiento con GPU habilitado
"""

import os
import sys
import subprocess

# Configurar para GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def restart_processing():
    """Reinicia el procesamiento con GPU"""
    print("🚀 REINICIANDO PROCESAMIENTO CON GPU")
    print("=" * 50)
    
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130.mp4",
        "--frame-processor", "face_swapper",  # Solo face_swapper para estabilidad
        "--gpu-memory-wait", "45",
        "--max-memory", "8",
        "--execution-threads", "24",
        "--temp-frame-quality", "90",
        "--execution-provider", "cuda,cpu",
        "--keep-fps"
    ]
    
    try:
        print("🔄 Ejecutando con GPU...")
        result = subprocess.run(command, timeout=3600)  # 1 hora timeout
        if result.returncode == 0:
            print("✅ Procesamiento completado con GPU")
            return True
        else:
            print("❌ Error en procesamiento")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Timeout")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

if __name__ == "__main__":
    restart_processing()
'''
    
    with open('restart_gpu_processing.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Script de reinicio creado: restart_gpu_processing.py")
    return True

def main():
    """Función principal"""
    print("🚀 SOLUCIONANDO PROBLEMA DE GPU")
    print("=" * 60)
    
    # Instalar libcufft
    if not install_cufft():
        print("❌ Error instalando libcufft")
        return False
    
    # Verificar GPU después del fix
    if not verify_gpu_after_fix():
        print("❌ GPU no disponible después del fix")
        return False
    
    # Crear script de reinicio
    if not create_restart_script():
        print("❌ Error creando script de reinicio")
        return False
    
    print("\n🎉 ¡PROBLEMA DE GPU SOLUCIONADO!")
    print("=" * 60)
    print("✅ libcufft instalado")
    print("✅ GPU verificada")
    print("✅ Script de reinicio creado")
    print("\n🚀 Para reiniciar el procesamiento con GPU:")
    print("python restart_gpu_processing.py")
    print("\n💡 Ahora debería usar GPU en lugar de CPU")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 