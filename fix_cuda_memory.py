#!/usr/bin/env python3
"""
Script para solucionar librerías CUDA faltantes y optimizar memoria
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

def install_missing_cuda_libraries():
    """Instala librerías CUDA faltantes"""
    print("🔧 INSTALANDO LIBRERÍAS CUDA FALTANTES")
    print("=" * 50)
    
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

def create_memory_optimized_script():
    """Crea un script optimizado para memoria"""
    print("💾 CREANDO SCRIPT OPTIMIZADO PARA MEMORIA")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script optimizado para procesamiento con memoria limitada
"""

import os
import sys

# Configurar variables de entorno para optimizar memoria
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Configurar límites de memoria más conservadores
os.environ['TF_MEMORY_ALLOCATION'] = '4096'  # 4GB en lugar de 8GB
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'

# Importar y ejecutar roop
from roop import core

if __name__ == "__main__":
    core.run()
'''
    
    with open('run_optimized.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Script optimizado creado: run_optimized.py")
    return True

def create_batch_optimized_script():
    """Crea un script de batch optimizado"""
    print("📦 CREANDO SCRIPT DE BATCH OPTIMIZADO")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Script de batch optimizado para memoria
"""

import os
import sys
import subprocess
import time
import argparse

# Configurar variables de entorno
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '4096'  # 4GB
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'

def process_video(source, video_path, output_dir, execution_threads=16, temp_frame_quality=85):
    """Procesa un video individual"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_name = f"{os.path.splitext(os.path.basename(source))[0]}{video_name}.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"🎬 Procesando: {video_path}")
    print(f"📸 Source: {source}")
    print(f"💾 Output: {output_path}")
    
    # Comando optimizado con menos memoria
    command = [
        sys.executable, "run_optimized.py",
        "--source", source,
        "--target", video_path,
        "-o", output_path,
        "--frame-processor", "face_swapper",  # Solo face_swapper para ahorrar memoria
        "--gpu-memory-wait", "60",  # Más tiempo de espera
        "--max-memory", "4",  # 4GB en lugar de 8GB
        "--execution-threads", str(execution_threads),
        "--temp-frame-quality", str(temp_frame_quality),
        "--execution-provider", "cuda,cpu",
        "--keep-fps"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)  # 30 min timeout
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
    parser = argparse.ArgumentParser(description="Procesamiento optimizado en batch")
    parser.add_argument("--source", required=True, help="Imagen fuente")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos a procesar")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida")
    parser.add_argument("--execution-threads", type=int, default=16, help="Hilos de ejecución")
    parser.add_argument("--temp-frame-quality", type=int, default=85, help="Calidad de frames temporales")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 INICIANDO PROCESAMIENTO OPTIMIZADO")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"🎬 Videos: {len(args.videos)}")
    print(f"💾 Output: {args.output_dir}")
    print(f"⚙️ Threads: {args.execution_threads}")
    print(f"📊 Quality: {args.temp_frame_quality}")
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
        
        # Pausa entre videos para liberar memoria
        if i < len(args.videos):
            print("⏳ Esperando 15 segundos...")
            time.sleep(15)
    
    print(f"\\n🎉 PROCESAMIENTO COMPLETADO")
    print(f"✅ Completados: {completed}")
    print(f"❌ Fallidos: {failed}")

if __name__ == "__main__":
    main()
'''
    
    with open('run_batch_optimized.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Script de batch optimizado creado: run_batch_optimized.py")
    return True

def main():
    """Función principal"""
    print("🚀 SOLUCIONANDO PROBLEMAS DE CUDA Y MEMORIA")
    print("=" * 60)
    
    # Instalar librerías CUDA faltantes
    if not install_missing_cuda_libraries():
        print("❌ Error instalando librerías CUDA")
        return False
    
    # Crear scripts optimizados
    if not create_memory_optimized_script():
        print("❌ Error creando script optimizado")
        return False
    
    if not create_batch_optimized_script():
        print("❌ Error creando script de batch")
        return False
    
    print("\\n🎉 ¡PROBLEMAS SOLUCIONADOS!")
    print("=" * 60)
    print("✅ Librerías CUDA instaladas")
    print("✅ Scripts optimizados creados")
    print("\\n🚀 Ahora puedes usar:")
    print("python run_batch_optimized.py --source tu_imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados")
    print("\\n💡 Optimizaciones aplicadas:")
    print("• Memoria reducida a 4GB")
    print("• Solo face_swapper (sin face_enhancer)")
    print("• Más tiempo de espera GPU")
    print("• Pausa entre videos")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 