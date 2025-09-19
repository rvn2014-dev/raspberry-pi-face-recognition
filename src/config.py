# src/config.py
import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Face detection settings
    FACE_DETECTION_METHOD = 'hog'  # 'hog' or 'cnn'
    FACE_RECOGNITION_TOLERANCE = 0.6
    SCALE_FACTOR = 0.25
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Display settings
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    # Voice notification settings
    ENABLE_VOICE_NOTIFICATIONS = True
    VOICE_ENGINE = 'espeak'  # 'pyttsx3' or 'espeak' (espeak is more reliable on Pi)
    VOICE_RATE = 150  # Words per minute
    VOICE_VOLUME = 0.9  # Volume level (0.0 to 1.0)
    VOICE_LANGUAGE = 'en'  # Language code
    RECOGNITION_COOLDOWN = 5  # Seconds to wait before announcing same person again
    
    # Model file
    ENCODINGS_FILE = os.path.join(MODELS_DIR, 'face_encodings.pickle')
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)