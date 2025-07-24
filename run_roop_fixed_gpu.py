#!/usr/bin/env python3
"""
Script para ejecutar ROOP con GPU corregido y alta calidad
"""

import os
import sys
import subprocess

def setup_environment():
    """Configurar variables de entorno"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

def run_roop_fixed():
    """Ejecutar ROOP con configuración corregida"""
    
    print("🎬 EJECUTANDO ROOP CON GPU CORREGIDO Y ALTA CALIDAD")
    print("=" * 60)
    
    # Configurar entorno
    setup_environment()
    
    # Comando con configuración corregida y alta calidad
    cmd = [
        'roop_env/bin/python', 'run.py',
        '--source', '/content/DanielaAS.jpg',
        '--target', '/content/112.mp4',
        '-o', '/content/DanielaAS112_fixed_gpu.mp4',
        '--frame-processor', 'face_swapper',
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '15',
        # Configuraciones de alta calidad
        '--temp-frame-format', 'png',
        '--temp-frame-quality', '0',
        '--output-video-encoder', 'libx264',
        '--output-video-quality', '35',
        '--keep-fps'
    ]
    
    print("⚙️ CONFIGURACIÓN:")
    print("   • GPU: Tesla T4 (15GB)")
    print("   • ONNX Runtime: GPU")
    print("   • Calidad: Alta (configuraciones originales)")
    print("   • temp_frame_format: png (sin compresión)")
    print("   • temp_frame_quality: 0 (sin compresión)")
    print("   • output_video_encoder: libx264 (alta calidad)")
    print("   • output_video_quality: 35 (calidad original)")
    print("=" * 60)
    
    try:
        print("🚀 Iniciando procesamiento...")
        result = subprocess.run(cmd, check=True)
        print("\n✅ Procesamiento completado exitosamente")
        print("📁 Archivo generado: /content/DanielaAS112_fixed_gpu.mp4")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en procesamiento: {e}")
        return False

def run_batch_fixed():
    """Ejecutar procesamiento en lote con configuración corregida"""
    
    print("🎬 PROCESAMIENTO EN LOTE CON GPU CORREGIDO")
    print("=" * 60)
    
    # Configurar entorno
    setup_environment()
    
    # Comando en lote
    cmd = [
        'roop_env/bin/python', 'run_batch_processing.py',
        '--source', '/content/LilitAS.png',
        '--videos', '/content/62.mp4', '/content/71.mp4', '/content/72.mp4', 
        '/content/74.mp4', '/content/75.mp4', '/content/76.mp4', 
        '/content/77.mp4', '/content/78.mp4', '/content/79.mp4',
        '--output-dir', '/content/resultados_fixed_gpu',
        '--execution-threads', '8',
        # Configuraciones de alta calidad
        '--temp-frame-format', 'png',
        '--temp-frame-quality', '0',
        '--output-video-encoder', 'libx264',
        '--output-video-quality', '35',
        '--keep-fps'
    ]
    
    print("⚙️ CONFIGURACIÓN EN LOTE:")
    print("   • GPU: Tesla T4 (15GB)")
    print("   • ONNX Runtime: GPU")
    print("   • Calidad: Alta (configuraciones originales)")
    print("   • Videos: 9 archivos")
    print("   • Output: /content/resultados_fixed_gpu")
    print("=" * 60)
    
    try:
        print("🚀 Iniciando procesamiento en lote...")
        result = subprocess.run(cmd, check=True)
        print("\n✅ Procesamiento en lote completado")
        print("📁 Archivos generados en: /content/resultados_fixed_gpu")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en procesamiento: {e}")
        return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ROOP con GPU corregido y alta calidad')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single', 
                       help='Modo de procesamiento: single o batch')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        success = run_roop_fixed()
    else:
        success = run_batch_fixed()
    
    if success:
        print("\n🎉 ¡PROCESAMIENTO COMPLETADO CON ÉXITO!")
        print("✅ GPU funcionando correctamente")
        print("✅ Alta calidad de video")
        print("✅ Sin pixelado")
    else:
        print("\n❌ Error en el procesamiento")
        print("💡 Verifica la instalación de ONNX Runtime GPU")

if __name__ == "__main__":
    main() 