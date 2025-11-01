# 🚗 Pothole Detection and Danger Assessment System

A real-time pothole detection system using deep learning for semantic segmentation. This application provides image, video, and live webcam analysis with danger level classification and dimension estimation.

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

- **🖼️ Image Analysis**: Upload images to detect potholes with bounding boxes and danger classification
- **🎥 Video Processing**: Analyze video files for pothole detection frame-by-frame
- **📹 Real-time Webcam Detection**: Live pothole detection using your webcam
- **📏 Dimension Estimation**: Calculate approximate pothole dimensions
- **⚠️ Danger Classification**: Automatic classification into 4 danger levels (Critical, High, Medium, Low)
- **🎨 Interactive GUI**: User-friendly PyQt5 desktop application
- **📊 MLflow Integration**: Experiment tracking and model versioning

## 🏗️ Architecture

The system uses state-of-the-art semantic segmentation models:

- **DeepLabV3+** with ResNet50 backbone
- **U-Net** with ResNet34 backbone
- **FPN** (Feature Pyramid Network) with EfficientNet-B0
- **Custom U-Net** architecture

## 📋 Requirements

- Python 3.9 or 3.10
- CUDA-capable GPU (optional, but recommended)
- Webcam (for real-time detection)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 Dependencies

```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
opencv-python==4.8.0.76
Pillow==10.0.0
albumentations==1.3.1
segmentation-models-pytorch==0.3.3
PyQt5==5.15.9
mlflow==2.8.0
efficientnet-pytorch
timm
```

## 🎮 Usage

### Running the Application

```bash
python app.py
```

### Application Modes

#### 1. **Image Mode**
- Click "Upload Image"
- Select an image file (PNG, JPG, JPEG)
- View detection results with:
  - Segmentation mask overlay
  - Bounding boxes
  - Danger level classification
  - Dimension estimates

#### 2. **Video Mode**
- Click "Upload Video"
- Select a video file (MP4, AVI, MOV)
- Process and view frame-by-frame detection
- Export processed video

#### 3. **Webcam Mode**
- Click "Start Webcam"
- Real-time pothole detection
- Live danger assessment
- Click "Stop Webcam" to end

## 📊 Model Training

The project includes a Jupyter notebook (`iheb.ipynb`) for training custom models:

```bash
jupyter notebook iheb.ipynb
```

### Training Features:
- Multiple architecture comparison
- Data augmentation with Albumentations
- MLflow experiment tracking
- Model checkpointing
- Loss and metric visualization

## 🗂️ Project Structure

```
pothole-detection/
├── .idea/                  # PyCharm IDE settings
├── .venv/                  # Virtual environment
├── mlflow_tracking/        # MLflow tracking data
├── mlruns/                 # MLflow experiment runs
├── models/                 # Trained model checkpoints
├── ui/                     # UI components
├── utils/                  # Utility functions
│   ├── preprocessing.py    # Image preprocessing
│   ├── danger_assessment.py # Danger classification
│   └── detector.py         # Detection logic
├── videos/                 # Sample videos
├── app.py                  # Main application
├── config.py               # Configuration settings
├── iheb.ipynb             # Training notebook
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
class Config:
    MODEL_PATH = 'models/checkpoints/deeplabv3plus_best.pth'
    INPUT_SIZE = (256, 256)
    NUM_CLASSES = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Danger thresholds (area percentage)
    DANGER_THRESHOLDS = {
        'critical': 60,  # > 60%
        'high': 40,      # 40-60%
        'medium': 20,    # 20-40%
        'low': 0         # < 20%
    }
```

## 🎯 Danger Classification Criteria

| Level | Area Coverage | Color | Action |
|-------|--------------|-------|--------|
| 🔴 **Critical** | > 60% | Red | Immediate repair required |
| 🟠 **High** | 40-60% | Orange | Urgent attention needed |
| 🟡 **Medium** | 20-40% | Yellow | Schedule maintenance |
| 🟢 **Low** | < 20% | Green | Monitor condition |

## 📈 Model Performance

| Model | IoU | Dice Score | Accuracy |
|-------|-----|------------|----------|
| DeepLabV3+ (ResNet50) | 0.87 | 0.93 | 96.2% |
| U-Net (ResNet34) | 0.85 | 0.91 | 95.8% |
| FPN (EfficientNet-B0) | 0.83 | 0.90 | 94.5% |
| Custom U-Net | 0.81 | 0.88 | 93.7% |

## 🔧 Troubleshooting

### PyQt5 DLL Error (Windows)

If you encounter `DLL load failed` error:

```bash
pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip
pip install PyQt5==5.15.9
```

Install Visual C++ Redistributables: [Download](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### CUDA Out of Memory

Reduce batch size in `config.py` or use CPU:

```python
DEVICE = 'cpu'
```

### Webcam Not Detected

Check camera permissions and ensure no other application is using it.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

*Initial work* -(https://github.com/medihebhichri)

## 🙏 Acknowledgments

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) for pretrained models
- [Albumentations](https://albumentations.ai/) for data augmentation
- [MLflow](https://mlflow.org/) for experiment tracking
- PyQt5 community for GUI framework

## 📧 Contact

For questions or suggestions, please open an issue or contact:
- Email: mediheb@esprit.tn
- GitHub:(https://github.com/medihebhichri)

## 🔮 Future Enhancements

- [ ] Mobile app integration
- [ ] GPS location tagging
- [ ] Cloud deployment
- [ ] Multi-class detection (cracks, road damage types)
- [ ] Depth estimation using stereo cameras
- [ ] Road condition reporting system
- [ ] Integration with municipal maintenance systems

---

**⭐ If you find this project useful, please consider giving it a star!**
