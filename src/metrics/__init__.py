from .eyes_rub import detect_rubbing
from .head_tilt import detect_head_tilt
from .blinking import detect_blink

__version__ = '0.0.3'

# from detectors import *
__all__ = ['detect_rubbing', 'detect_head_tilt', 'detect_blink']
