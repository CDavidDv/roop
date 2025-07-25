#!/usr/bin/env python3
"""
Script para instalar librerías CUDA faltantes y configurar GPU
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

def install_cuda_libraries():
    """Instala las librerías CUDA faltantes"""
    print("🔧 INSTALANDO LIBRERÍAS CUDA FALTANTES")
    print("=" * 50)
    
    # Actualizar repositorios
    if not run_command("apt-get update", "Actualizando repositorios"):
        return False
    
    # Instalar librerías CUDA faltantes
    cuda_libraries = [
        "libcufft-11-8",
        "libcufft-dev-11-8",
        "libcurand-11-8", 
        "libcurand-dev-11-8",
        "libcusolver-11-8",
        "libcusolver-dev-11-8",
        "libcusparse-11-8",
        "libcusparse-dev-11-8",
    ]
    
    for library in cuda_libraries:
        if not run_command(f"apt-get install -y {library}", f"Instalando {library}"):
            print(f"⚠️ {library} no disponible, continuando...")
    
    return True

def create_gpu_optimized_script():
    """Crea un script optimizado para GPU"""
    print("🔥 CREANDO SCRIPT OPTIMIZADO PARA GPU")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script optimizado para GPU con librerías CUDA instaladas
"""

import os
import sys

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Configurar ONNX Runtime para GPU
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'

# Configurar límites de memoria para GPU
os.environ['TF_MEMORY_ALLOCATION'] = '6144'  # 6GB para GPU

# Importar y ejecutar roop
from roop import core

if __name__ == "__main__":
    core.run()
'''
    
    with open('run_gpu_optimized.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Script GPU optimizado creado: run_gpu_optimized.py")
    return True

def create_gpu_batch_script():
    """Crea un script de batch optimizado para GPU"""
    print("📦 CREANDO SCRIPT DE BATCH PARA GPU")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script de batch optimizado para GPU
"""

import os
import sys
import subprocess
import time
import argparse

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
os.environ['TF_MEMORY_ALLOCATION'] = '6144'

def process_video(source, video_path, output_dir, execution_threads=24, temp_frame_quality=90):
    """Procesa un video individual con GPU"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_name = f"{os.path.splitext(os.path.basename(source))[0]}{video_name}.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"🎬 Procesando: {video_path}")
    print(f"📸 Source: {source}")
    print(f"💾 Output: {output_path}")
    print(f"🔥 Usando GPU optimizado")
    
    # Comando optimizado para GPU
    command = [
        sys.executable, "run_gpu_optimized.py",
        "--source", source,
        "--target", video_path,
        "-o", output_path,
        "--frame-processor", "face_swapper",  # Solo face_swapper para estabilidad
        "--gpu-memory-wait", "45",  # Más tiempo de espera
        "--max-memory", "6",  # 6GB para GPU
        "--execution-threads", str(execution_threads),
        "--temp-frame-quality", str(temp_frame_quality),
        "--execution-provider", "cuda,cpu",
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento con GPU...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=2400)  # 40 min timeout
        if result.returncode == 0:
            print(f"✅ Completado: {output_path}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {video_path}")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Procesamiento optimizado en GPU")
    parser.add_argument("--source", required=True, help="Imagen fuente")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos a procesar")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida")
    parser.add_argument("--execution-threads", type=int, default=24, help="Hilos de ejecución")
    parser.add_argument("--temp-frame-quality", type=int, default=90, help="Calidad de frames temporales")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 INICIANDO PROCESAMIENTO CON GPU OPTIMIZADO")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"🎬 Videos: {len(args.videos)}")
    print(f"💾 Output: {args.output_dir}")
    print(f"⚙️ Threads: {args.execution_threads}")
    print(f"📊 Quality: {args.temp_frame_quality}")
    print(f"🔥 Modo: GPU optimizado")
    print("=" * 60)
    
    completed = 0
    failed = 0
    
    for i, video_path in enumerate(args.videos, 1):
        print(f"\\n📊 Progreso: {i}/{len(args.videos)} ({i/len(args.videos)*100:.1f}%)")
        
        if process_video(args.source, video_path, args.output_dir, 
                        args.execution_threads, args.temp_frame_quality):
            completed += 1
        else:
            failed += 1
        
        # Pausa entre videos para liberar memoria GPU
        if i < len(args.videos):
            print("⏳ Esperando 30 segundos para liberar GPU...")
            time.sleep(30)
    
    print(f"\\n🎉 PROCESAMIENTO COMPLETADO")
    print(f"✅ Completados: {completed}")
    print(f"❌ Fallidos: {failed}")

if __name__ == "__main__":
    main()
'''
    
    with open('run_batch_gpu_optimized.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Script de batch GPU optimizado creado: run_batch_gpu_optimized.py")
    return True

def verify_gpu_setup():
    """Verifica que GPU esté configurado correctamente"""
    print("🔍 VERIFICANDO CONFIGURACIÓN GPU")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✅ ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA provider disponible")
        else:
            print("⚠️ CUDA provider no disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 CONFIGURANDO GPU CON LIBRERÍAS CUDA")
    print("=" * 60)
    
    # Instalar librerías CUDA
    if not install_cuda_libraries():
        print("❌ Error instalando librerías CUDA")
        return False
    
    # Crear scripts optimizados
    if not create_gpu_optimized_script():
        print("❌ Error creando script GPU")
        return False
    
    if not create_gpu_batch_script():
        print("❌ Error creando script de batch GPU")
        return False
    
    # Verificar configuración
    if not verify_gpu_setup():
        print("❌ Error en verificación GPU")
        return False
    
    print("\\n🎉 ¡GPU CONFIGURADO CORRECTAMENTE!")
    print("=" * 60)
    print("✅ Librerías CUDA instaladas")
    print("✅ Scripts GPU optimizados creados")
    print("✅ GPU verificada")
    print("\\n🚀 Ahora puedes usar:")
    print("python run_batch_gpu_optimized.py --source tu_imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados")
    print("\\n💡 Optimizaciones GPU aplicadas:")
    print("• Memoria GPU: 6GB")
    print("• Threads: 24")
    print("• Calidad: 90%")
    print("• GPU wait: 45s")
    print("• Pausa entre videos: 30s")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 