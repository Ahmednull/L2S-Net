import pathlib
from pathlib import Path
from typing import Union,Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer
import gdown
import os
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class L2CSConfig:
    """Configuration for L2CS model paths and parameters"""
    # Model folder ID for all models
    FOLDER_ID = "17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"
    
    # Direct file IDs for specific models
    MODEL_FILE_IDS = {
        'gaze360': "18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ"  # Direct link to gaze360 model
    }
    
    # Local paths where models will be stored
    MODEL_PATHS = {
        'gaze360': "models/L2CSNet_gaze360.pkl",
        'mpiigaze': "models/L2CSNet_mpiigaze.pkl"
    }
    
    @classmethod
    def initialize(cls, model_type: str = 'gaze360'):
        """
        Initialize model directories and download if needed.
        For gaze360, downloads single file. For others, downloads from folder.
        """
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Check if model exists
            model_path = cls.MODEL_PATHS.get(model_type)
            if not model_path:
                raise ValueError(f"Unknown model type: {model_type}")
                
            if not os.path.exists(model_path):
                logger.info(f"Downloading L2CS {model_type} model to {model_path}...")
                
                try:
                    # If it's gaze360, download single file
                    if model_type == 'gaze360' and model_type in cls.MODEL_FILE_IDS:
                        gdown.download(
                            id=cls.MODEL_FILE_IDS[model_type],
                            output=model_path,
                            quiet=False,
                            use_cookies=False
                        )
                    # Otherwise download from folder
                    else:
                        gdown.download_folder(
                            id=cls.FOLDER_ID,
                            output=str(Path(model_path).parent),
                            quiet=False,
                            use_cookies=False
                        )
                    
                    # Check if download was successful
                    if not os.path.exists(model_path):
                        raise RuntimeError(f"Model file not found after download")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {str(e)}")
                
            logger.info("L2CS model initialization complete.")
            
        except Exception as e:
            logger.error(f"Failed to initialize L2CS: {str(e)}")
            raise
        


class Pipeline:

    def __init__(
        self, 
        weights: Optional[Path] = None,
        model_type: str = 'gaze360',
        arch: str = 'ResNet50',
        device: str = 'cpu',
        include_detector: bool = True,
        confidence_threshold: float = 0.5
        ):
        
        # Initialize model paths and check dependencies
        L2CSConfig.initialize(model_type)
        
        # Use provided weights path or default to downloaded model
        if weights is None:
            weights = Path(L2CSConfig.MODEL_PATHS[model_type])
        
        # Parse device string
        self.device_str = device
        if device == 'cpu':
            self.device = torch.device('cpu')
            self.gpu_id = None
        else:
            if ':' in device:
                self.gpu_id = int(device.split(':')[1])
            else:
                self.gpu_id = 0
            self.device = torch.device(f'cuda:{self.gpu_id}')

        # Create RetinaFace if requested
        if include_detector:
            try:
                from face_detection import RetinaFace
                if self.device_str == 'cpu':
                    self.detector = RetinaFace()
                else:
                    self.detector = RetinaFace(gpu_id=self.gpu_id)
                self.detector_available = True
            except ImportError:
                logger.warning("face_detection package not available. Face detection disabled.")
                self.detector_available = False
                include_detector = False

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        self.model = getArch(arch, 90)
        
        # Load weights
        try:
            self.model.load_state_dict(torch.load(str(weights), map_location=self.device))
        except Exception as e:
            raise RuntimeError(f"Failed to load L2CS weights from {weights}: {str(e)}")
            
        self.model.to(self.device)
        self.model.eval()

        # Initialize other components
        if self.include_detector:
            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)

    def step(self, frame: np.ndarray):
        """Process a single frame"""
        with torch.no_grad():
            if self.include_detector and self.detector_available:
                faces = self.detector(frame)
                
                if faces is not None:
                    face_imgs = []
                    bboxes = []
                    landmarks = []
                    scores = []
                    
                    for box, landmark, score in faces:
                        if score < self.confidence_threshold:
                            continue
                            
                        # Extract face region
                        x_min = max(int(box[0]), 0)
                        y_min = max(int(box[1]), 0)
                        x_max = int(box[2])
                        y_max = int(box[3])
                        
                        img = frame[y_min:y_max, x_min:x_max]
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        face_imgs.append(img)
                        
                        bboxes.append(box)
                        landmarks.append(landmark)
                        scores.append(score)
                        
                    if face_imgs:  # If faces were detected
                        pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                    else:
                        pitch = np.empty((0,1))
                        yaw = np.empty((0,1))
                        bboxes = np.array([])
                        landmarks = np.array([])
                        scores = np.array([])
                else:
                    pitch = np.empty((0,1))
                    yaw = np.empty((0,1))
                    bboxes = np.array([])
                    landmarks = np.array([])
                    scores = np.array([])
            else:
                pitch, yaw = self.predict_gaze(frame)
                bboxes = np.array([])
                landmarks = np.array([])
                scores = np.array([])
                
            return GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.array(bboxes) if len(bboxes) > 0 else np.empty((0,4)),
                landmarks=np.array(landmarks) if len(landmarks) > 0 else np.empty((0,5,2)),
                scores=np.array(scores) if len(scores) > 0 else np.empty(0)
            )

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict gaze angles from input frame(s)"""
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame.to(self.device)
        else:
            raise RuntimeError("Invalid dtype for input")

        # Get predictions
        gaze_pitch, gaze_yaw = self.model(img)
        
        # Convert predictions
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0
        
        return pitch_predicted, yaw_predicted
