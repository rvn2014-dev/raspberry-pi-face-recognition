# src/voice_notifier.py
import pyttsx3
import subprocess
import threading
import time
from config import Config
import logging

class VoiceNotifier:
    def __init__(self):
        self.config = Config()
        self.engine = None
        self.last_recognition_times = {}  # Track when each person was last announced
        self.setup_logging()
        self.initialize_tts()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'{self.config.LOGS_DIR}/voice_notifications.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_tts(self):
        """Initialize text-to-speech engine"""
        if not self.config.ENABLE_VOICE_NOTIFICATIONS:
            self.logger.info("Voice notifications disabled in config")
            return
        
        try:
            if self.config.VOICE_ENGINE == 'pyttsx3':
                self.engine = pyttsx3.init()
                
                # Set basic properties only - avoid voice selection issues
                try:
                    self.engine.setProperty('rate', self.config.VOICE_RATE)
                    self.engine.setProperty('volume', self.config.VOICE_VOLUME)
                    self.logger.info(f"Set voice rate to {self.config.VOICE_RATE} and volume to {self.config.VOICE_VOLUME}")
                except Exception as e:
                    self.logger.warning(f"Could not set voice properties: {e}")
                
                # Skip voice selection entirely to avoid the error
                self.logger.info("Using default system voice to avoid voice selection issues")
                
                # Test the engine with a simple phrase
                try:
                    test_text = "Voice system ready"
                    self.engine.say(test_text)
                    self.engine.runAndWait()
                    self.logger.info("pyttsx3 TTS engine test successful")
                except Exception as e:
                    self.logger.warning(f"TTS engine test failed: {e}")
                    raise Exception(f"pyttsx3 test failed: {e}")
                
                self.logger.info("pyttsx3 TTS engine initialized successfully")
            
            elif self.config.VOICE_ENGINE == 'espeak':
                # Test espeak availability
                result = subprocess.run(['which', 'espeak'], capture_output=True)
                if result.returncode != 0:
                    raise Exception("espeak not found")
                
                # Test espeak with a simple command
                try:
                    test_result = subprocess.run(
                        ['espeak', '-s', str(self.config.VOICE_RATE), 'Voice system ready'], 
                        check=True, capture_output=True, timeout=5
                    )
                    self.logger.info("espeak TTS engine test successful")
                except subprocess.CalledProcessError as e:
                    raise Exception(f"espeak test failed: {e}")
                except subprocess.TimeoutExpired:
                    self.logger.warning("espeak test timed out but engine seems available")
                
                self.logger.info("espeak TTS engine available and working")
                self.engine = None  # espeak doesn't need an engine object
            
            else:
                raise Exception(f"Unknown voice engine: {self.config.VOICE_ENGINE}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.VOICE_ENGINE} TTS engine: {e}")
            
            # Try to fallback to espeak if pyttsx3 failed
            if self.config.VOICE_ENGINE == 'pyttsx3':
                self.logger.info("Attempting fallback to espeak...")
                try:
                    result = subprocess.run(['which', 'espeak'], capture_output=True)
                    if result.returncode == 0:
                        # Test espeak
                        subprocess.run(['espeak', '--version'], check=True, capture_output=True)
                        self.logger.info("Successfully switched to espeak engine")
                        self.config.VOICE_ENGINE = 'espeak'
                        self.engine = None
                        return
                except Exception as fallback_error:
                    self.logger.error(f"Fallback to espeak also failed: {fallback_error}")
            
            # If all else fails, disable voice notifications
            self.logger.error("No working TTS engine available - voice notifications disabled")
            self.config.ENABLE_VOICE_NOTIFICATIONS = False
            self.engine = None
    
    def should_announce(self, person_name):
        """Check if enough time has passed to announce this person again"""
        if person_name == "Unknown":
            return False
        
        current_time = time.time()
        last_time = self.last_recognition_times.get(person_name, 0)
        
        if current_time - last_time >= self.config.RECOGNITION_COOLDOWN:
            self.last_recognition_times[person_name] = current_time
            return True
        
        return False
    
    def speak_pyttsx3(self, text):
        """Speak using pyttsx3 engine"""
        try:
            if self.engine is None:
                self.logger.warning("pyttsx3 engine not initialized, cannot speak")
                return
            
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"pyttsx3 speech error: {e}")
            # Try to reinitialize engine once
            try:
                self.logger.info("Attempting to reinitialize pyttsx3 engine")
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', self.config.VOICE_RATE)
                self.engine.setProperty('volume', self.config.VOICE_VOLUME)
                self.engine.say(text)
                self.engine.runAndWait()
                self.logger.info("Successfully reinitialized and spoke")
            except Exception as reinit_error:
                self.logger.error(f"Reinitialize failed: {reinit_error}")
                self.logger.info("Switching to espeak fallback")
                self.speak_espeak(text)
    
    def speak_espeak(self, text):
        """Speak using espeak"""
        try:
            cmd = [
                'espeak', 
                f'-s{self.config.VOICE_RATE}',
                f'-a{int(self.config.VOICE_VOLUME * 100)}',
                f'-v{self.config.VOICE_LANGUAGE}',
                text
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"espeak speech error: {e}")
        except Exception as e:
            self.logger.error(f"espeak error: {e}")
    
    def announce_recognition(self, person_names):
        """Announce recognized faces (non-blocking)"""
        if not self.config.ENABLE_VOICE_NOTIFICATIONS or not person_names:
            return
        
        # Filter names that should be announced
        names_to_announce = [name for name in person_names if self.should_announce(name)]
        
        if not names_to_announce:
            return
        
        # Create announcement text
        if len(names_to_announce) == 1:
            text = f"Hello {names_to_announce[0]}"
        elif len(names_to_announce) == 2:
            text = f"Hello {names_to_announce[0]} and {names_to_announce[1]}"
        else:
            text = f"Hello {', '.join(names_to_announce[:-1])} and {names_to_announce[-1]}"
        
        # Speak in a separate thread to avoid blocking
        thread = threading.Thread(target=self._speak_threaded, args=(text,))
        thread.daemon = True
        thread.start()
        
        self.logger.info(f"Announced: {text}")
    
    def _speak_threaded(self, text):
        """Thread-safe speech function"""
        try:
            if self.config.VOICE_ENGINE == 'pyttsx3' and self.engine:
                self.speak_pyttsx3(text)
            elif self.config.VOICE_ENGINE == 'espeak':
                self.speak_espeak(text)
            elif self.config.VOICE_ENGINE == 'pyttsx3':
                # pyttsx3 was selected but engine failed, try espeak
                self.logger.info("pyttsx3 failed, trying espeak fallback")
                self.speak_espeak(text)
            else:
                self.logger.warning("No TTS engine available")
        except Exception as e:
            self.logger.error(f"Speech thread error: {e}")
            # Last resort: try espeak directly
            try:
                self.speak_espeak(text)
            except:
                self.logger.error("All speech methods failed")
    
    def test_speech(self):
        """Test the speech system"""
        if not self.config.ENABLE_VOICE_NOTIFICATIONS:
            print("Voice notifications are disabled in config")
            return
        
        test_message = "Voice notification system is working correctly"
        print(f"Testing speech: {test_message}")
        self._speak_threaded(test_message)