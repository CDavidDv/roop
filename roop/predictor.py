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
    if DISABLE_NSFW_CHECK or get_predictor() is None:
        return False
    
    try:
        image = Image.fromarray(target_frame)
        image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
        views = numpy.expand_dims(image, axis=0)
        _, probability = get_predictor().predict(views)[0]
        return probability > MAX_PROBABILITY
    except Exception as e:
        print(f"[PREDICTOR] Error en predict_frame: {e}")
        return False


def predict_image(target_path: str) -> bool:
    if DISABLE_NSFW_CHECK or get_predictor() is None:
        return False
    
    try:
        return opennsfw2.predict_image(target_path) > MAX_PROBABILITY
    except Exception as e:
        print(f"[PREDICTOR] Error en predict_image: {e}")
        return False


def predict_video(target_path: str) -> bool:
    if DISABLE_NSFW_CHECK or get_predictor() is None:
        print("[PREDICTOR] Verificación NSFW desactivada automáticamente")
        return False
    
    try:
        _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
        return any(probability > MAX_PROBABILITY for probability in probabilities)
    except Exception as e:
        print(f"[PREDICTOR] Error en predict_video: {e}")
        print("[PREDICTOR] Desactivando verificación NSFW automáticamente")
        return False
