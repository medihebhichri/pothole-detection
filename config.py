import torch


class Config:
    # UPDATE THIS PATH to where your model is located
    MODEL_PATH = r'/models/checkpoints\deeplabv3plus_best.pth'  # Change this!

    INPUT_SIZE = (256, 256)
    NUM_CLASSES = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    DANGER_LEVELS = {
        'CRITICAL': {'min': 60, 'color': (255, 0, 0), 'action': 'Immediate repair'},
        'HIGH': {'min': 40, 'color': (255, 165, 0), 'action': 'Urgent repair 48-72h'},
        'MEDIUM': {'min': 20, 'color': (255, 255, 0), 'action': 'Scheduled repair 1-2 weeks'},
        'LOW': {'min': 0, 'color': (0, 255, 0), 'action': 'Routine maintenance'}
    }