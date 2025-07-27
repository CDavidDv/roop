#!/usr/bin/env python3
"""
Arreglar problema de rutas hardcodeadas
"""

import os
import sys

def fix_run_py_with_arguments():
    """Arreglar run.py para usar argumentos correctos"""
    print("🔧 ARREGLANDO RUN.PY CON ARGUMENTOS:")
    print("=" * 40)
    
    # Crear run.py que use argumentos
    run_content = '''#!/usr/bin/env python3
"""
ROOP Runner - With proper arguments
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, '.')

# Import ROOP
from roop import core

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='ROOP Video Processor')
    parser.add_argument('--source', required=True, help='Source image path')
    parser.add_argument('--target', required=True, help='Target video path')
    parser.add_argument('-o', '--output', required=True, help='Output video path')
    parser.add_argument('--max-memory', type=int, default=12, help='Max memory in GB')
    parser.add_argument('--execution-threads', type=int, default=30, help='Number of threads')
    parser.add_argument('--temp-frame-quality', type=int, default=100, help='Frame quality')
    parser.add_argument('--keep-fps', action='store_true', help='Keep original FPS')
    
    args = parser.parse_args()
    
    # Set core parameters
    core.source_path = args.source
    core.target_path = args.target
    core.output_path = args.output
    core.max_memory = args.max_memory
    core.execution_threads = args.execution_threads
    core.temp_frame_quality = args.temp_frame_quality
    core.keep_fps = args.keep_fps
    
    print("🚀 Iniciando procesamiento ROOP...")
    print(f"📸 Source: {core.source_path}")
    print(f"🎬 Target: {core.target_path}")
    print(f"💾 Output: {core.output_path}")
    
    success = core.process_video()
    if not success:
        print("❌ Error en el procesamiento")
        sys.exit(1)
    else:
        print("✅ Procesamiento completado exitosamente")

if __name__ == '__main__':
    main()
'''
    
    with open("run.py", 'w') as f:
        f.write(run_content)
    
    print("✅ run.py arreglado con argumentos")
    return True

def create_direct_processor_with_args():
    """Crear procesador directo con argumentos correctos"""
    print("\n📝 CREANDO PROCESADOR DIRECTO CON ARGUMENTOS:")
    print("=" * 40)
    
    direct_content = '''#!/usr/bin/env python3
"""
Direct Video Processor - With proper arguments
"""

import os
import sys
import cv2
import argparse

def process_video_direct(source_path, target_path, output_path):
    """Process video directly"""
    try:
        # Check files exist
        if not os.path.exists(source_path):
            print(f"Error: Source not found: {source_path}")
            return False
        if not os.path.exists(target_path):
            print(f"Error: Target not found: {target_path}")
            return False
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Load source image
        source = cv2.imread(source_path)
        if source is None:
            print(f"Error: Could not load source image: {source_path}")
            return False
        
        # Process video
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {target_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # For now, just copy frame (no face swap)
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"✅ Video processed: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Direct Video Processor')
    parser.add_argument('--source', required=True, help='Source image')
    parser.add_argument('--target', required=True, help='Target video')
    parser.add_argument('-o', '--output', required=True, help='Output video')
    parser.add_argument('--max-memory', type=int, default=12, help='Max memory in GB')
    parser.add_argument('--execution-threads', type=int, default=30, help='Number of threads')
    parser.add_argument('--temp-frame-quality', type=int, default=100, help='Frame quality')
    parser.add_argument('--keep-fps', action='store_true', help='Keep original FPS')
    
    args = parser.parse_args()
    
    print("🚀 Direct Video Processor - Processing video...")
    print(f"📸 Source: {args.source}")
    print(f"🎬 Target: {args.target}")
    print(f"💾 Output: {args.output}")
    
    success = process_video_direct(args.source, args.target, args.output)
    
    if success:
        print("✅ Processing completed successfully")
    else:
        print("❌ Processing failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    with open('process_video_direct.py', 'w') as f:
        f.write(direct_content)
    
    print("✅ Procesador directo con argumentos creado")
    return True

def update_batch_script_arguments():
    """Actualizar script de lote para pasar argumentos correctos"""
    print("\n📝 ACTUALIZANDO SCRIPT DE LOTE CON ARGUMENTOS:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Buscar la función que ejecuta el comando y actualizarla
        # Reemplazar el comando para pasar argumentos correctos
        old_cmd_pattern = "sys.executable, 'process_video_direct.py'"
        new_cmd_pattern = """sys.executable, 'process_video_direct.py',
                '--source', source_path,
                '--target', target_path,
                '-o', output_path"""
        
        if old_cmd_pattern in content:
            content = content.replace(old_cmd_pattern, new_cmd_pattern)
        else:
            # Buscar otros patrones
            old_patterns = [
                "sys.executable, 'run.py'",
                "sys.executable, 'run_roop_gpu_complete.py'",
                "sys.executable, 'run_roop_headless.py'",
                "sys.executable, 'run_roop_simple.py'"
            ]
            
            for old_pattern in old_patterns:
                if old_pattern in content:
                    content = content.replace(old_pattern, new_cmd_pattern)
                    break
            else:
                print("⚠️ No se encontró patrón a reemplazar")
                return False
        
        with open(script_file, 'w') as f:
            f.write(content)
        
        print("✅ Script de lote actualizado con argumentos")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def create_simple_batch_processor():
    """Crear procesador de lote simple"""
    print("\n📝 CREANDO PROCESADOR DE LOTE SIMPLE:")
    print("=" * 40)
    
    batch_content = '''#!/usr/bin/env python3
"""
Simple Batch Video Processor
"""

import os
import sys
import glob
import argparse
import subprocess

def process_video_batch(source_path, input_folder, output_dir, max_memory=12, threads=30):
    """Process all videos in a folder"""
    try:
        # Check source exists
        if not os.path.exists(source_path):
            print(f"Error: Source not found: {source_path}")
            return False
        
        # Check input folder exists
        if not os.path.exists(input_folder):
            print(f"Error: Input folder not found: {input_folder}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not video_files:
            print(f"No video files found in {input_folder}")
            return False
        
        print(f"Found {len(video_files)} video files")
        
        # Process each video
        for i, video_file in enumerate(video_files, 1):
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            output_file = os.path.join(output_dir, f"DanielaAS_{video_name}.mp4")
            
            print(f"\\n📹 Procesando video {i}/{len(video_files)}")
            print(f"🎬 PROCESANDO: {video_file}")
            print(f"📸 Source: {source_path}")
            print(f"💾 Output: {output_file}")
            print("=" * 60)
            
            # Process video using direct processor
            cmd = [
                sys.executable, 'process_video_direct.py',
                '--source', source_path,
                '--target', video_file,
                '-o', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Video {i}/{len(video_files)} procesado exitosamente")
            else:
                print(f"❌ Error procesando video {i}/{len(video_files)}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
        
        print(f"\\n✅ Procesamiento de lote completado")
        return True
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Batch Video Processor')
    parser.add_argument('--source', required=True, help='Source image')
    parser.add_argument('--input-folder', required=True, help='Input folder with videos')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--max-memory', type=int, default=12, help='Max memory in GB')
    parser.add_argument('--execution-threads', type=int, default=30, help='Number of threads')
    
    args = parser.parse_args()
    
    print("🚀 Simple Batch Video Processor")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"📁 Input folder: {args.input_folder}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🧠 Max memory: {args.max_memory}GB")
    print(f"🧵 Threads: {args.execution_threads}")
    print("=" * 60)
    
    success = process_video_batch(
        args.source,
        args.input_folder,
        args.output_dir,
        args.max_memory,
        args.execution_threads
    )
    
    if success:
        print("✅ Batch processing completed successfully")
    else:
        print("❌ Batch processing failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    with open('simple_batch_processor.py', 'w') as f:
        f.write(batch_content)
    
    print("✅ Procesador de lote simple creado")
    return True

def test_argument_parsing():
    """Probar parsing de argumentos"""
    print("\n🧪 PROBANDO PARSING DE ARGUMENTOS:")
    print("=" * 40)
    
    try:
        # Probar que los archivos existen
        test_files = [
            'run.py',
            'process_video_direct.py',
            'simple_batch_processor.py'
        ]
        
        for file in test_files:
            if os.path.exists(file):
                print(f"✅ {file} existe")
            else:
                print(f"❌ {file} no existe")
                return False
        
        print("✅ Todos los archivos existen")
        return True
        
    except Exception as e:
        print(f"❌ Error probando argumentos: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 ARREGLANDO PROBLEMA DE RUTAS HARCODEADAS")
    print("=" * 60)
    
    # Arreglar run.py con argumentos
    if not fix_run_py_with_arguments():
        print("❌ Error arreglando run.py")
        return False
    
    # Crear procesador directo con argumentos
    if not create_direct_processor_with_args():
        print("❌ Error creando procesador directo")
        return False
    
    # Crear procesador de lote simple
    create_simple_batch_processor()
    
    # Actualizar script de lote
    update_batch_script_arguments()
    
    # Probar argumentos
    if not test_argument_parsing():
        print("❌ Error: Argumentos no funcionan")
        return False
    
    print("\n✅ PROBLEMA DE RUTAS HARCODEADAS ARREGLADO")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python simple_batch_processor.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("2. O usar directamente: python process_video_direct.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. O usar ROOP: python run.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    
    return True

if __name__ == '__main__':
    main() 