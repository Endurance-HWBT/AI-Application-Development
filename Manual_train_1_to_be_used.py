from ultralytics import YOLO
from pathlib import Path
import sys

# ---------------- CONFIG ----------------
RAW_DIR = Path(r"C:\Users\Piyush\PycharmProjects\PythonProject3\data_new\Apple")
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 224
PROJECT_DIR = Path("runs")
RUN_NAME = "apple_train"  # optional, leave blank to auto-generate


# ----------------------------------------

def find_latest_weights(project_dir: Path):
    """Find the latest training run and return path to best.pt or last.pt"""
    if not project_dir.exists():
        sys.exit(f"❌ Project directory {project_dir} does not exist.")

    run_folders = [d for d in project_dir.iterdir() if d.is_dir()]
    if not run_folders:
        sys.exit(f"❌ No training runs found in {project_dir}.")

    latest_run = max(run_folders, key=lambda p: p.stat().st_mtime)
    best_weights = latest_run / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"⚠️ best.pt not found in {latest_run}, using last.pt instead.")
        best_weights = latest_run / "weights" / "last.pt"

    if not best_weights.exists():
        sys.exit(f"❌ No weight files found in {latest_run}/weights.")

    print(f"✅ Using weights from: {best_weights}")
    return best_weights


def train_model(data_dir: Path, epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    # Load base YOLOv8 classification model
    model = YOLO("yolov8n-cls.pt")
    print(f"🚀 Starting training for {epochs} epochs...")

    # Train
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
    )

    # Find latest weights
    return find_latest_weights(PROJECT_DIR)


if __name__ == "__main__":
    # Train model and get latest saved weights
    weights_path = train_model(RAW_DIR)

    # Load the model for inference
    model = YOLO(weights_path)

    # Test a single image (replace with your test image path)
    test_image = RAW_DIR / "test_image.jpg"
    if test_image.exists():
        results = model.predict(str(test_image))
        results.show()
        print(f"Predicted class: {results[0].names[results[0].probs.argmax()]}, "
              f"confidence: {results[0].probs.max() * 100:.2f}%")
    else:
        print("No test image found. Training weights are ready for inference.")
