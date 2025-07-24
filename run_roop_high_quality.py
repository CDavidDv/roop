#!/usr/bin/env python3
"""
Script para ejecutar ROOP con configuraciones originales para alta calidad
"""

import os
import sys
import subprocess

def run_roop_high_quality(source_path, target_path, output_path):
    """
    Ejecutar ROOP con configuraciones originales para alta calidad
    
    Configuraciones originales del repositorio:
    - temp_frame_format: 'png' (sin compresión)
    - temp_frame_quality: 0 (sin compresión)
    - output_video_encoder: 'libx264' (alta calidad)
    - output_video_quality: 35 (calidad original)
    """
    
    # Configurar variables de entorno para GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Comando con configuraciones originales para alta calidad
    cmd = [
        'roop_env/bin/python', 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper',
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '15',
        # Configuraciones originales para alta calidad
        '--temp-frame-format', 'png',  # Sin compresión
        '--temp-frame-quality', '0',   # Sin compresión
        '--output-video-encoder', 'libx264',  # Encoder de alta calidad
        '--output-video-quality', '35',  # Calidad original del repositorio
        '--keep-fps'
    ]
    
    print("🎬 EJECUTANDO ROOP CON ALTA CALIDAD")
    print("=" * 50)
    print(f"📸 Source: {source_path}")
    print(f"🎥 Target: {target_path}")
    print(f"💾 Output: {output_path}")
    print("\n⚙️ CONFIGURACIONES DE CALIDAD:")
    print("   • temp_frame_format: png (sin compresión)")
    print("   • temp_frame_quality: 0 (sin compresión)")
    print("   • output_video_encoder: libx264 (alta calidad)")
    print("   • output_video_quality: 35 (calidad original)")
    print("   • execution_provider: cuda (GPU)")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Procesamiento completado con alta calidad")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en procesamiento: {e}")
        return False

def run_batch_high_quality(source_path, video_paths, output_dir):
    """
    Ejecutar procesamiento en lote con alta calidad
    """
    
    print("🎬 PROCESAMIENTO EN LOTE CON ALTA CALIDAD")
    print("=" * 50)
    
    for i, video_path in enumerate(video_paths, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        output_name = f"{source_name}_{video_name}.mp4"
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\n🎬 PROCESANDO {i}/{len(video_paths)}: {video_name}")
        print(f"📸 Source: {source_name}")
        print(f"💾 Output: {output_name}")
        
        success = run_roop_high_quality(source_path, video_path, output_path)
        
        if success:
            print(f"✅ {video_name} completado")
        else:
            print(f"❌ Error en {video_name}")
    
    print("\n🎉 ¡Procesamiento en lote completado!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ROOP con alta calidad')
    parser.add_argument('--source', required=True, help='Ruta de la imagen fuente')
    parser.add_argument('--target', help='Ruta del video objetivo (para procesamiento individual)')
    parser.add_argument('-o', '--output', help='Ruta de salida (para procesamiento individual)')
    parser.add_argument('--videos', nargs='+', help='Lista de videos para procesamiento en lote')
    parser.add_argument('--output-dir', help='Directorio de salida para procesamiento en lote')
    
    args = parser.parse_args()
    
    if args.target and args.output:
        # Procesamiento individual
        run_roop_high_quality(args.source, args.target, args.output)
    elif args.videos and args.output_dir:
        # Procesamiento en lote
        run_batch_high_quality(args.source, args.videos, args.output_dir)
    else:
        print("❌ Uso:")
        print("   Procesamiento individual: --source imagen.jpg --target video.mp4 -o salida.mp4")
        print("   Procesamiento en lote: --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados") 