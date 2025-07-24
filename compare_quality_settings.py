#!/usr/bin/env python3
"""
Script para comparar configuraciones de calidad de ROOP
"""

def print_quality_comparison():
    """Mostrar comparación de configuraciones de calidad"""
    
    print("📊 COMPARACIÓN DE CONFIGURACIONES DE CALIDAD")
    print("=" * 60)
    
    # Configuraciones que causan baja calidad
    low_quality = {
        'temp_frame_format': 'jpg',
        'temp_frame_quality': 100,
        'output_video_encoder': 'h264_nvenc',
        'output_video_quality': 100,
        'description': 'Baja calidad (pixelado)'
    }
    
    # Configuraciones originales del repositorio (alta calidad)
    high_quality = {
        'temp_frame_format': 'png',
        'temp_frame_quality': 0,
        'output_video_encoder': 'libx264',
        'output_video_quality': 35,
        'description': 'Alta calidad (original)'
    }
    
    print(f"{'Configuración':<20} {'Baja Calidad':<20} {'Alta Calidad':<20}")
    print("-" * 60)
    print(f"{'temp_frame_format':<20} {low_quality['temp_frame_format']:<20} {high_quality['temp_frame_format']:<20}")
    print(f"{'temp_frame_quality':<20} {low_quality['temp_frame_quality']:<20} {high_quality['temp_frame_quality']:<20}")
    print(f"{'output_video_encoder':<20} {low_quality['output_video_encoder']:<20} {high_quality['output_video_encoder']:<20}")
    print(f"{'output_video_quality':<20} {low_quality['output_video_quality']:<20} {high_quality['output_video_quality']:<20}")
    print("-" * 60)
    print(f"{'Descripción':<20} {low_quality['description']:<20} {high_quality['description']:<20}")
    
    print("\n🔍 EXPLICACIÓN:")
    print("=" * 40)
    print("❌ CONFIGURACIONES QUE CAUSAN BAJA CALIDAD:")
    print("   • temp_frame_format: jpg (con compresión)")
    print("   • temp_frame_quality: 100 (máxima compresión)")
    print("   • output_video_encoder: h264_nvenc (hardware, menos calidad)")
    print("   • output_video_quality: 100 (puede causar artefactos)")
    
    print("\n✅ CONFIGURACIONES ORIGINALES (ALTA CALIDAD):")
    print("   • temp_frame_format: png (sin compresión)")
    print("   • temp_frame_quality: 0 (sin compresión)")
    print("   • output_video_encoder: libx264 (software, alta calidad)")
    print("   • output_video_quality: 35 (calidad original del repositorio)")

def print_commands():
    """Mostrar comandos con diferentes configuraciones"""
    
    print("\n🚀 COMANDOS CON DIFERENTES CONFIGURACIONES:")
    print("=" * 60)
    
    print("\n❌ COMANDO CON BAJA CALIDAD (NO USAR):")
    print("roop_env/bin/python run.py \\")
    print("  --source imagen.jpg \\")
    print("  --target video.mp4 \\")
    print("  -o salida_baja_calidad.mp4 \\")
    print("  --frame-processor face_swapper \\")
    print("  --execution-provider cuda \\")
    print("  --temp-frame-format jpg \\")
    print("  --temp-frame-quality 100 \\")
    print("  --output-video-encoder h264_nvenc \\")
    print("  --output-video-quality 100 \\")
    print("  --keep-fps")
    
    print("\n✅ COMANDO CON ALTA CALIDAD (RECOMENDADO):")
    print("roop_env/bin/python run.py \\")
    print("  --source imagen.jpg \\")
    print("  --target video.mp4 \\")
    print("  -o salida_alta_calidad.mp4 \\")
    print("  --frame-processor face_swapper \\")
    print("  --execution-provider cuda \\")
    print("  --temp-frame-format png \\")
    print("  --temp-frame-quality 0 \\")
    print("  --output-video-encoder libx264 \\")
    print("  --output-video-quality 35 \\")
    print("  --keep-fps")
    
    print("\n🎬 COMANDO EN LOTE CON ALTA CALIDAD:")
    print("roop_env/bin/python run_batch_processing.py \\")
    print("  --source imagen.jpg \\")
    print("  --videos video1.mp4 video2.mp4 \\")
    print("  --output-dir resultados \\")
    print("  --temp-frame-format png \\")
    print("  --temp-frame-quality 0 \\")
    print("  --output-video-encoder libx264 \\")
    print("  --output-video-quality 35 \\")
    print("  --keep-fps")

def print_script_usage():
    """Mostrar uso del script optimizado"""
    
    print("\n📝 USO DEL SCRIPT OPTIMIZADO:")
    print("=" * 40)
    
    print("\n1. Procesamiento individual:")
    print("   python run_roop_high_quality.py --source imagen.jpg --target video.mp4 -o salida.mp4")
    
    print("\n2. Procesamiento en lote:")
    print("   python run_roop_high_quality.py --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados")
    
    print("\n3. Verificar configuración:")
    print("   python compare_quality_settings.py")

def main():
    """Función principal"""
    print("🎬 CONFIGURACIONES DE CALIDAD PARA ROOP")
    print("=" * 60)
    
    print_quality_comparison()
    print_commands()
    print_script_usage()
    
    print("\n💡 RECOMENDACIÓN:")
    print("=" * 30)
    print("✅ Usa siempre las configuraciones originales del repositorio:")
    print("   • temp_frame_format: png")
    print("   • temp_frame_quality: 0")
    print("   • output_video_encoder: libx264")
    print("   • output_video_quality: 35")
    print("\n❌ Evita configuraciones que causan pixelado:")
    print("   • temp_frame_format: jpg")
    print("   • temp_frame_quality: 100")
    print("   • output_video_encoder: h264_nvenc")
    print("   • output_video_quality: 100")

if __name__ == "__main__":
    main() 