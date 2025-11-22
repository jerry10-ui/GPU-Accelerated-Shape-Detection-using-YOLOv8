"""
train_shapes.py
GPU-accelerated YOLOv8 training script for geometric shape detection
"""

from ultralytics import YOLO
import torch
import os


def main():
    # === 1️⃣ Verify GPU and environment ===
    print("🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available — training will run on CPU (slower)")

    # === 2️⃣ Define dataset and model paths ===
    project_dir = r"C:\Coding\GPU Project"
    data_yaml = os.path.join(project_dir, "dataset", "data.yaml")
    model_name = "yolov8n.pt"  # nano model (fast and light for testing)

    # === 3️⃣ Initialize and train the model ===
    print("\n🚀 Starting YOLOv8 training...")
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        project=project_dir,
        name="shape_train_run",
        verbose=True
    )

    # === 4️⃣ Evaluate and show results ===
    print("\n📊 Training complete! Summary:")
    print(results)

    # === 5️⃣ Optional: Run predictions on validation or test images ===
    test_dir = os.path.join(project_dir, "dataset", "test", "images")
    if os.path.exists(test_dir):
        print("\n🎯 Running prediction on test images...")
        model.predict(source=test_dir, save=True, show=False)
        print(f"✅ Predictions saved inside: {os.path.join(project_dir, 'shape_train_run', 'detect')}")
    else:
        print("⚠️ Test folder not found — skipping predictions.")

    print("\n✅ Training finished successfully!")


# === ✅ Windows-safe entry point ===
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # required for Windows multiprocessing
    main()
