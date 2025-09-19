# src/main.py
import cv2
import argparse
import os
from face_recognizer import FaceRecognizer
from face_trainer import FaceTrainer
from camera_handler import CameraHandler
from voice_notifier import VoiceNotifier
from config import Config
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def train_faces(images_path):
    """Train face recognition model"""
    logger = setup_logging()
    logger.info("Starting face training...")
    
    trainer = FaceTrainer()
    trainer.train_from_images(images_path)
    
    logger.info("Training completed successfully!")

def run_recognition(use_pi_camera=True, headless=False, save_images=False, enable_voice=True):
    """Run real-time face recognition"""
    logger = setup_logging()
    logger.info("Starting face recognition...")
    
    # Set environment variables for headless operation
    if headless:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['DISPLAY'] = ':99'
    
    # Initialize components
    config = Config()
    camera = CameraHandler(use_pi_camera=use_pi_camera)
    recognizer = FaceRecognizer()
    
    # Initialize voice notifier
    voice_notifier = None
    if enable_voice:
        voice_notifier = VoiceNotifier()
        # Test voice system on startup
        if voice_notifier.config.ENABLE_VOICE_NOTIFICATIONS:
            logger.info("Voice notifications enabled")
        else:
            logger.info("Voice notifications disabled")
    
    frame_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = camera.read_frame()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Recognize faces
            frame, names = recognizer.recognize_faces(frame)
            
            # Voice notifications for recognized faces
            if voice_notifier and names:
                # Filter out "Unknown" faces for voice announcements
                known_names = [name for name in names if name != "Unknown"]
                if known_names:
                    voice_notifier.announce_recognition(known_names)
            
            # Log recognized faces
            if names:
                unique_names = list(set(names))
                logger.info(f"Recognized: {', '.join(unique_names)}")
            
            # Save images if requested
            if save_images and names and any(name != "Unknown" for name in names):
                filename = f"logs/recognized_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Saved frame to {filename}")
            
            if not headless:
                try:
                    # Display frame
                    cv2.imshow('Face Recognition', frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    logger.error(f"Display error: {e}")
                    logger.info("Switching to headless mode...")
                    headless = True
            else:
                # In headless mode, run for a limited time or until interrupted
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(f"Processed {frame_count} frames")
                
                # You can add conditions to break the loop in headless mode
                # For example, after processing a certain number of frames
                # if frame_count > 1000:
                #     break
    
    except KeyboardInterrupt:
        logger.info("Recognition stopped by user")
    except Exception as e:
        logger.error(f"Error during recognition: {e}")
    finally:
        camera.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("Face recognition stopped")

def test_voice():
    """Test voice notification system"""
    logger = setup_logging()
    logger.info("Testing voice notification system...")
    
    voice_notifier = VoiceNotifier()
    voice_notifier.test_speech()

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Face Recognition')
    parser.add_argument('--mode', choices=['train', 'recognize', 'test-voice'], required=True,
                       help='Mode: train, recognize, or test-voice')
    parser.add_argument('--images-path', type=str, default='data/training_images',
                       help='Path to training images directory')
    parser.add_argument('--camera', choices=['pi', 'usb'], default='pi',
                       help='Camera type: pi or usb')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI display (for SSH/remote access)')
    parser.add_argument('--save-images', action='store_true',
                       help='Save frames when faces are recognized')
    parser.add_argument('--no-voice', action='store_true',
                       help='Disable voice notifications')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_faces(args.images_path)
    elif args.mode == 'recognize':
        use_pi_camera = args.camera == 'pi'
        enable_voice = not args.no_voice
        run_recognition(use_pi_camera, args.headless, args.save_images, enable_voice)
    elif args.mode == 'test-voice':
        test_voice()

if __name__ == '__main__':
    main()