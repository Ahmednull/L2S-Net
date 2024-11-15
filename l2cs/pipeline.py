import pathlib
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


class L2CSConfig:
    """Configuration for L2CS model paths and parameters"""
    # Model folder ID from L2CS-Net Google Drive
    FOLDER_ID = "17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"
    
    # Model file names in the folder
    MODEL_FILES = {
        'gaze360': "L2CSNet_gaze360.pkl",  # Update with actual filename
        'mpiigaze': "L2CSNet_mpiigaze.pkl"  # Update with actual filename
    }
    
    # Local paths where models will be stored
    MODEL_PATHS = {
        'gaze360': "models/L2CSNet_gaze360.pkl",
        'mpiigaze': "models/L2CSNet_mpiigaze.pkl"
    }
    
    @classmethod
    def initialize(cls, model_type: str = 'gaze360'):
        """Initialize model directories and download if needed"""
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Check if model exists
            model_path = cls.MODEL_PATHS.get(model_type)
            model_file = cls.MODEL_FILES.get(model_type)
            
            if model_path and not os.path.exists(model_path):
                print(f"Downloading L2CS {model_type} model to {model_path}...")
                
                try:
                    # Download from Google Drive folder
                    gdown.download_folder(
                        id=cls.FOLDER_ID,
                        output=str(pathlib.Path(model_path).parent),
                        quiet=False,
                        use_cookies=False
                    )
                    
                    # Check if download was successful
                    if not os.path.exists(model_path):
                        raise RuntimeError(f"Model file {model_file} not found in downloaded folder")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {str(e)}")
                
            print("L2CS model initialization complete.")
            
        except Exception as e:
            print(f"Failed to initialize L2CS: {str(e)}")
            raise



class Pipeline:

    def __init__(
        self, 
        weights: Optional[pathlib.Path] = None,
        model_type: str = 'gaze360',
        arch: str = 'ResNet50',
        device: str = 'cpu',
        include_detector: bool = True,
        confidence_threshold: float = 0.5
        ):

        # Initialize model paths and download if needed
        L2CSConfig.initialize(model_type)
        
        # Use provided weights path or default to downloaded model
        if weights is None:
            weights = pathlib.Path(L2CSConfig.MODEL_PATHS[model_type])
            
        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        self.model = getArch(arch, 90)
        
        # Load weights
        try:
            self.model.load_state_dict(torch.load(self.weights, map_location=device))
        except Exception as e:
            raise RuntimeError(f"Failed to load L2CS weights from {self.weights}: {str(e)}")
            
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested
        if self.include_detector:
            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = [idx for idx in range(90)]
            self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None: 
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:

                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            pitch, yaw = self.predict_gaze(frame)

        # Save data
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes),
            landmarks=np.stack(landmarks),
            scores=np.stack(scores)
        )

        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Predict 
        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        return pitch_predicted, yaw_predicted
