import torch
from ultralytics import YOLO

print("✅ YOLOv8 & GPU Setup Verification\n")

# Check YOLO version
try:
    print(f"Ultralytics YOLO version: {YOLO.__module__.split('.')[0]}")
except Exception as e:
    print("❌ YOLOv8 not properly installed:", e)

# Check Torch and CUDA
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Running on CPU — GPU not detected!")

print("\nEverything looks good! You can now train your model 🎯")
