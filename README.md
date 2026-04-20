# GPU-Accelerated Real-Time Shape Detection using YOLOv8

This project implements a real-time geometric shape detection system using **YOLOv8**, **PyTorch**, and **CUDA GPU acceleration**.  
It includes an interactive **Tkinter dashboard** that supports:

- Live webcam detection
- Image and Video file detection
- Shape draw-pad (user drawing) recognition
- Text-to-Speech feedback
- Real-time GPU load & FPS display

The project demonstrates the power of **GPU parallel computing** for deep learning inference.

---

## Features
- Real-time detection using YOLOv8
- GPU-accelerated training and inference (CUDA-enabled)
- Tkinter GUI with:
  - Webcam mode  
  - Image and Video upload mode  
  - Shape drawing pad  
- Text-to-speech output (pyttsx3)
- GPU utilization and FPS monitoring (GPUtil)
- Custom YOLOv8 model trained on 10 geometric shapes

---

## Project Structure
```
GPU-Shape-Detection-YOLOv8/
├── dataset/ (download from the link given below and add to the project folder)
├── dashboard.py
├── train_main.py
├── verify_setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset
- Source: https://universe.roboflow.com/samarths-new-workspace/geometric-shape
- Contains **10 classes**:
  Circle, Triangle, Square, Rectangle, Star, Hexagon, Ellipse, Rhombus, Pentagon, Quatrefoil
- Annotations in YOLO format
- Dataset is *not included* in the repo due to size (ignored by .gitignore)

---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/jerry10-ui/GPU-Shape-Detection-YOLOv8.git
cd GPU-Shape-Detection-YOLOv8
```

### 2. Create Conda environment
```bash
conda create -n yolov8 python=3.11 -y
conda activate yolov8
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## GPU Setup Verification
```bash
python verify_setup.py
```

---

## Training the Model
Replace "project_dir = r"ADD PATH TO THE EXTRACTED REPOSITORY"" with the path to your cloned repository in train_main.py
```bash
python train_main.py
```

The best model weights will be saved in:
```
shape_train_main/weights/best.pt
```

---

## Running the Application
### Launch GUI:
```bash
python dashboard.py
```

Choose from:
- Webcam detection  
- Image and Video file detection  
- Draw-pad mode  

---

## Results
- **FPS:** 30–40 FPS real-time inference
- **mAP@50:** ~96%
- **Training Speedup:** 70–80% faster vs CPU
- Highly accurate detection across all 10 geometric shapes

---

## Applications
- Real-time vision systems  
- Industrial automation  
- Robotics and navigation  
- AI-powered educational tools  
- Visual inspection and analysis  

---
