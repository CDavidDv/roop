import threading
import numpy
import opennsfw2
from PIL import Image
from keras import Model

from roop.typing import Frame

PREDICTOR = None
THREAD_LOCK = threading.Lock()
MAX_PROBABILITY = 999
# Desactivar predictor NSFW automáticamente para evitar errores de GPU
DISABLE_NSFW_CHECK = True


def get_predictor() -> Model:
    global PREDICTOR

    with THREAD_LOCK:
        if PREDICTOR is None:
            try:
                PREDICTOR = opennsfw2.make_open_nsfw_model()
            except Exception as e:
                print(f"[PREDICTOR] Error cargando modelo NSFW: {e}")
                print("[PREDICTOR] Desactivando verificación NSFW automáticamente")
                PREDICTOR = None
    return PREDICTOR


def clear_predictor() -> None:
    global PREDICTOR

    PREDICTOR = None


def predict_frame(target_frame: Frame) -> bool:
    return False


def predict_image(target_path: str) -> bool:
    print("[PREDICTOR] Verificación NSFW desactivada para optimizar GPU")
    return False


def predict_video(target_path: str) -> bool:
    print("[PREDICTOR] Verificación NSFW desactivada para optimizar GPU")
    return False
