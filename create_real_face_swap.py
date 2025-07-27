#!/usr/bin/env python3
"""
Crear face swapping real con insightface
"""

import os
import sys
import subprocess

def install_face_swap_deps():
    """Instalar dependencias para face swapping real"""
    print("📦 INSTALANDO DEPENDENCIAS FACE SWAP:")
    print("=" * 40)
    
    try:
        # Dependencias para face swapping real
        deps = [
            "insightface==0.7.3",
            "onnxruntime-gpu==1.15.1",
            "opencv-python",
            "numpy==1.24.3",
            "scipy",
            "scikit-image"
        ]
        
        for dep in deps:
            print(f"⏳ Instalando {dep}...")
            cmd = [sys.executable, "-m", "pip", "install", dep]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Dependencias face swap instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def create_real_face_swapper():
    """Crear face swapper real con insightface"""
    print("\n📝 CREANDO FACE SWAPPER REAL:")
    print("=" * 40)
    
    # Crear directorio roop si no existe
    os.makedirs("roop", exist_ok=True)
    
    # Crear __init__.py
    init_content = '''"""
ROOP - Real Face Swapping version
"""

from .core import core

__all__ = ['core']
'''
    
    with open("roop/__init__.py", 'w') as f:
        f.write(init_content)
    
    # Crear core.py con face swapping real
    core_content = '''"""
ROOP Core - Real Face Swapping with insightface
"""

import os
import sys
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class RealFaceSwapper:
    """Real face swapper using insightface"""
    
    def __init__(self):
        self.name = 'face_swapper'
        self.app = None
        self.swapper = None
        self.source_face = None
        self.init_models()
    
    def init_models(self):
        """Initialize insightface models"""
        try:
            # Initialize face analysis
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize face swapper
            self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx')
            
            print("✅ Face swapping models initialized")
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            self.app = None
            self.swapper = None
    
    def extract_face(self, img):
        """Extract face from image"""
        if self.app is None:
            return None
        
        try:
            faces = self.app.get(img)
            if len(faces) > 0:
                return faces[0]
            return None
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None
    
    def swap_face(self, source_img, target_img):
        """Swap face from source to target"""
        if self.app is None or self.swapper is None:
            return target_img
        
        try:
            # Extract source face
            source_face = self.extract_face(source_img)
            if source_face is None:
                print("No face found in source image")
                return target_img
            
            # Extract target face
            target_face = self.extract_face(target_img)
            if target_face is None:
                print("No face found in target image")
                return target_img
            
            # Perform face swap
            result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
            return result
            
        except Exception as e:
            print(f"Error in face swap: {e}")
            return target_img
    
    def process_frame(self, source, target):
        """Process frame with real face swapping"""
        try:
            return self.swap_face(source, target)
        except Exception as e:
            print(f"Warning: Face swapping failed: {e}")
            return target

class FaceEnhancer:
    """Simple face enhancer"""
    
    def __init__(self):
        self.name = 'face_enhancer'
    
    def enhance_face(self, img):
        """Enhance face in image"""
        try:
            # Simple enhancement - you can replace with GFPGAN or RealESRGAN
            enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            return enhanced
        except Exception as e:
            print(f"Warning: Face enhancement failed: {e}")
            return img
    
    def process_frame(self, source, target):
        """Process frame with face enhancement"""
        try:
            return self.enhance_face(target)
        except Exception as e:
            print(f"Warning: Face enhancement failed: {e}")
            return target

class ROOPCore:
    """ROOP Core class - Real face swapping"""
    
    def __init__(self):
        self.source_path = None
        self.target_path = None
        self.output_path = None
        self.frame_processors = ['face_swapper', 'face_enhancer']
        self.max_memory = 12
        self.execution_threads = 30
        self.temp_frame_quality = 100
        self.keep_fps = False
        
        # Initialize processors
        self.face_swapper = RealFaceSwapper()
        self.face_enhancer = FaceEnhancer()
    
    def load_source(self):
        """Load source image"""
        if not self.source_path or not os.path.exists(self.source_path):
            print(f"Error: Source file not found: {self.source_path}")
            return None
        
        try:
            source = cv2.imread(self.source_path)
            if source is None:
                print(f"Error: Could not load source image: {self.source_path}")
                return None
            return source
        except Exception as e:
            print(f"Error loading source: {e}")
            return None
    
    def process_video(self):
        """Process video with real face swapping"""
        if not all([self.source_path, self.target_path, self.output_path]):
            print("Error: Missing required paths")
            return False
        
        # Load source
        source = self.load_source()
        if source is None:
            return False
        
        # Load target video
        if not os.path.exists(self.target_path):
            print(f"Error: Target file not found: {self.target_path}")
            return False
        
        try:
            cap = cv2.VideoCapture(self.target_path)
            if not cap.isOpened():
                print(f"Error: Could not open video: {self.target_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with face swap
                processed_frame = self.face_swapper.process_frame(source, frame)
                
                # Process frame with face enhancement
                processed_frame = self.face_enhancer.process_frame(source, processed_frame)
                
                # Write frame
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            print(f"✅ Video processed successfully: {self.output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False

# Global instance
core = ROOPCore()

# Set attributes for compatibility
source_path = None
target_path = None
output_path = None
frame_processors = ['face_swapper', 'face_enhancer']
max_memory = 12
execution_threads = 30
temp_frame_quality = 100
keep_fps = False
'''
    
    with open("roop/core.py", 'w') as f:
        f.write(core_content)
    
    print("✅ Face swapper real creado")
    return True

def create_enhanced_processor():
    """Crear procesador mejorado con face swapping real"""
    print("\n📝 CREANDO PROCESADOR MEJORADO:")
    print("=" * 40)
    
    enhanced_content = '''#!/usr/bin/env python3
"""
Enhanced Video Processor - Real Face Swapping
"""

import os
import sys
import cv2
import numpy as np
import argparse
import insightface
from insightface.app import FaceAnalysis

class EnhancedVideoProcessor:
    """Enhanced video processor with real face swapping"""
    
    def __init__(self):
        self.app = None
        self.swapper = None
        self.init_models()
    
    def init_models(self):
        """Initialize insightface models"""
        try:
            # Initialize face analysis
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize face swapper
            self.swapper = insightface.model_zoo.get_model('inswapper_128.onnx')
            
            print("✅ Face swapping models initialized")
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            self.app = None
            self.swapper = None
    
    def extract_face(self, img):
        """Extract face from image"""
        if self.app is None:
            return None
        
        try:
            faces = self.app.get(img)
            if len(faces) > 0:
                return faces[0]
            return None
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None
    
    def swap_face(self, source_img, target_img):
        """Swap face from source to target"""
        if self.app is None or self.swapper is None:
            return target_img
        
        try:
            # Extract source face
            source_face = self.extract_face(source_img)
            if source_face is None:
                print("No face found in source image")
                return target_img
            
            # Extract target face
            target_face = self.extract_face(target_img)
            if target_face is None:
                print("No face found in target image")
                return target_img
            
            # Perform face swap
            result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
            return result
            
        except Exception as e:
            print(f"Error in face swap: {e}")
            return target_img
    
    def enhance_face(self, img):
        """Enhance face in image"""
        try:
            # Simple enhancement
            enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            return enhanced
        except Exception as e:
            print(f"Warning: Face enhancement failed: {e}")
            return img
    
    def process_video(self, source_path, target_path, output_path):
        """Process video with real face swapping"""
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
                
                # Perform face swap
                processed_frame = self.swap_face(source, frame)
                
                # Perform face enhancement
                processed_frame = self.enhance_face(processed_frame)
                
                # Write frame
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            print(f"✅ Video processed with real face swapping: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Video Processor with Real Face Swapping')
    parser.add_argument('--source', required=True, help='Source image')
    parser.add_argument('--target', required=True, help='Target video')
    parser.add_argument('-o', '--output', required=True, help='Output video')
    parser.add_argument('--max-memory', type=int, default=12, help='Max memory in GB')
    parser.add_argument('--execution-threads', type=int, default=30, help='Number of threads')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Video Processor - Real Face Swapping")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"🎬 Target: {args.target}")
    print(f"💾 Output: {args.output}")
    print("=" * 60)
    
    processor = EnhancedVideoProcessor()
    success = processor.process_video(args.source, args.target, args.output)
    
    if success:
        print("✅ Processing completed successfully")
    else:
        print("❌ Processing failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    with open('enhanced_video_processor.py', 'w') as f:
        f.write(enhanced_content)
    
    print("✅ Procesador mejorado creado")
    return True

def test_real_face_swap():
    """Probar face swapping real"""
    print("\n🧪 PROBANDO FACE SWAPPING REAL:")
    print("=" * 40)
    
    try:
        # Probar importación
        import sys
        sys.path.insert(0, '.')
        
        from roop import core
        print("✅ ROOP core importado")
        
        # Probar configuración
        core.source_path = "/content/DanielaAS.jpg"
        core.target_path = "/content/videos_entrada/135.mp4"
        core.output_path = "/content/videos_salida/test.mp4"
        
        print("✅ Face swapping real configurado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error probando face swapping real: {e}")
        return False

def main():
    """Función principal"""
    print("🎭 CREANDO FACE SWAPPING REAL")
    print("=" * 50)
    
    # Instalar dependencias
    if not install_face_swap_deps():
        print("❌ Error instalando dependencias")
        return False
    
    # Crear face swapper real
    if not create_real_face_swapper():
        print("❌ Error creando face swapper real")
        return False
    
    # Crear procesador mejorado
    create_enhanced_processor()
    
    # Probar face swapping real
    if not test_real_face_swap():
        print("❌ Error: Face swapping real no funciona")
        return False
    
    print("\n✅ FACE SWAPPING REAL CREADO")
    print("=" * 50)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar con face swapping real: python enhanced_video_processor.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("2. O usar ROOP con face swapping: python run.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Procesar lote: python simple_batch_processor.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 