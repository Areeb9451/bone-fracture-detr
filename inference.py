import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "model"

ID2LABEL = {
    0: "bone-fracture",
    1: "angle",
    2: "fracture",
    3: "line",
    4: "messed_up_angle",
}

CONFIDENCE_THRESHOLD = 0.35


# -----------------------------
# Load model
# -----------------------------

device = torch.device("cpu")

processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Load image
# -----------------------------

image_path = input("Enter image path: ").strip()

image = Image.open(image_path).convert("RGB")


# -----------------------------
# Run inference
# -----------------------------

inputs = processor(images=image, return_tensors="pt")

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)


# -----------------------------
# Post-process predictions
# -----------------------------

target_sizes = torch.tensor(
    [[image.height, image.width]]
).to(device)

results = processor.post_process_object_detection(
    outputs,
    threshold=CONFIDENCE_THRESHOLD,
    target_sizes=target_sizes
)[0]


# -----------------------------
# Display predictions
# -----------------------------

print("\nPredictions:")

if len(results["scores"]) == 0:
    print("No detections found.")
else:
    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):
        label_id = label.item()
        confidence = score.item()

        box = [round(x, 2) for x in box.tolist()]

        class_name = ID2LABEL.get(
            label_id,
            f"unknown_class_{label_id}"
        )

        print(
            f"{class_name}: "
            f"{confidence:.2f} "
            f"box={box}"
        )