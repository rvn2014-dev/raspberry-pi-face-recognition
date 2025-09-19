# src/face_trainer.py
import os
import pickle
import face_recognition
import cv2
from config import Config
import logging

class FaceTrainer:
    def __init__(self):
        self.config = Config()
        self.known_encodings = []
        self.known_names = []
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'{self.config.LOGS_DIR}/training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def train_from_images(self, images_path):
        """Train the model from images in subdirectories"""
        self.logger.info("Starting face training...")
        
        for person_name in os.listdir(images_path):
            person_path = os.path.join(images_path, person_name)
            if not os.path.isdir(person_path):
                continue
            
            self.logger.info(f"Processing images for {person_name}")
            
            for image_name in os.listdir(person_path):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                image_path = os.path.join(person_path, image_name)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    # Use the first face found
                    encoding = encodings[0]
                    self.known_encodings.append(encoding)
                    self.known_names.append(person_name)
                    self.logger.info(f"Added encoding for {person_name} from {image_name}")
                else:
                    self.logger.warning(f"No face found in {image_path}")
        
        # Save encodings
        self.save_encodings()
        self.logger.info(f"Training completed. Total encodings: {len(self.known_encodings)}")
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        
        with open(self.config.ENCODINGS_FILE, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Encodings saved to {self.config.ENCODINGS_FILE}")