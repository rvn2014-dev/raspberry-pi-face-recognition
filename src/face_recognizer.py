# src/face_recognizer.py
import pickle
import face_recognition
import cv2
from config import Config
from face_detector import FaceDetector
import logging

class FaceRecognizer:
    def __init__(self):
        self.config = Config()
        self.face_detector = FaceDetector()
        self.known_encodings = []
        self.known_names = []
        self.setup_logging()
        self.load_encodings()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'{self.config.LOGS_DIR}/recognition.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_encodings(self):
        """Load face encodings from file"""
        try:
            with open(self.config.ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            self.logger.info(f"Loaded {len(self.known_encodings)} face encodings")
        except FileNotFoundError:
            self.logger.warning("No encodings file found. Please train the model first.")
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Detect faces
        face_locations = self.face_detector.detect_faces(frame)
        
        if not face_locations:
            return frame, []
        
        # Get encodings for detected faces
        face_encodings = self.face_detector.get_face_encodings(frame, face_locations)
        
        face_names = []
        
        # Compare with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding,
                tolerance=self.config.FACE_RECOGNITION_TOLERANCE
            )
            name = "Unknown"
            
            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
            
            face_names.append(name)
        
        # Draw rectangles and labels
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, self.config.FONT_SCALE, (255, 255, 255), 1)
        
        return frame, face_names