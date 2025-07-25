#!/usr/bin/env python3
"""
Script optimizado para CPU que evita problemas de CUDA
"""

import os
import sys

# Configurar para usar SOLO CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Deshabilitar GPU
os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'  # Solo CPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Configurar límites de memoria más conservadores
os.environ['TF_MEMORY_ALLOCATION'] = '2048'  # 2GB para CPU

# Importar y ejecutar roop
from roop import core

if __name__ == "__main__":
    core.run() 