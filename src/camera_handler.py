# src/camera_handler.py
import cv2
from picamera2 import Picamera2
from config import Config
import logging

class CameraHandler:
    def __init__(self, use_pi_camera=True):
        self.config = Config()
        self.use_pi_camera = use_pi_camera
        self.camera = None
        self.setup_logging()
        self.initialize_camera()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'{self.config.LOGS_DIR}/camera.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self):
        """Initialize camera based on type"""
        try:
            if self.use_pi_camera:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"format": 'XRGB8888', 
                          "size": (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT)}
                )
                self.camera.configure(config)
                self.camera.start()
                self.logger.info("Pi Camera initialized successfully")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
                self.camera.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
                self.logger.info("USB Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def read_frame(self):
        """Read frame from camera"""
        try:
            if self.use_pi_camera:
                frame = self.camera.capture_array()
                return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                return self.camera.read()
        except Exception as e:
            self.logger.error(f"Failed to read frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        try:
            if self.use_pi_camera:
                self.camera.stop()
            else:
                self.camera.release()
            self.logger.info("Camera released successfully")
        except Exception as e:
            self.logger.error(f"Failed to release camera: {e}")