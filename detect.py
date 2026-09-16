import cv2
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

device = torch.device("cpu")


# -----------------------------
# Load model
# -----------------------------

print("Loading model...")

processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("Model loaded.")


# -----------------------------
# Get image
# -----------------------------

image_path = input("Enter image path: ").strip()

image = Image.open(image_path).convert("RGB")

# Convert PIL → OpenCV
image_cv = cv2.cvtColor(
    __import__("numpy").array(image),
    cv2.COLOR_RGB2BGR
)


# -----------------------------
# Inference
# -----------------------------

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)


# -----------------------------
# Post-processing
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
# Draw detections
# -----------------------------

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"]
):

    score = score.item()
    label_id = label.item()

    x1, y1, x2, y2 = [
        int(x) for x in box.tolist()
    ]

    class_name = ID2LABEL.get(
        label_id,
        f"unknown_{label_id}"
    )

    text = f"{class_name} {score:.2f}"

    cv2.rectangle(
        image_cv,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image_cv,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    print(
        f"{class_name}: {score:.2f} "
        f"box={[x1, y1, x2, y2]}"
    )


# -----------------------------
# Save result
# -----------------------------

output_path = "result.jpg"

cv2.imwrite(output_path, image_cv)

print()
print(f"Result saved to: {output_path}")