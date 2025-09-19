# src/face_detector.py
import cv2
import face_recognition
import numpy as np
from config import Config
import logging

class FaceDetector:
    def __init__(self):
        self.config = Config()
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'{self.config.LOGS_DIR}/face_detection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_faces(self, frame):
        """Detect faces in a frame and return their locations"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.config.SCALE_FACTOR, 
                                fy=self.config.SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame, 
            model=self.config.FACE_DETECTION_METHOD
        )
        
        # Scale back up face locations
        face_locations = [(int(top/self.config.SCALE_FACTOR), 
                          int(right/self.config.SCALE_FACTOR),
                          int(bottom/self.config.SCALE_FACTOR), 
                          int(left/self.config.SCALE_FACTOR)) 
                         for (top, right, bottom, left) in face_locations]
        
        return face_locations
    
    def get_face_encodings(self, frame, face_locations):
        """Get face encodings for detected faces"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return encodings