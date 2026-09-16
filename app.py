import cv2
import torch
import gradio as gr
import numpy as np

from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


# ---------------------------------
# Configuration
# ---------------------------------

MODEL_PATH = "model"

ID2LABEL = {
    0: "bone-fracture",
    1: "angle",
    2: "fracture",
    3: "line",
    4: "messed_up_angle",
}

device = torch.device("cpu")


# ---------------------------------
# Load model ONCE
# ---------------------------------

print("Loading DETR model...")

processor = DetrImageProcessor.from_pretrained(MODEL_PATH)

model = DetrForObjectDetection.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ---------------------------------
# Prediction function
# ---------------------------------

def detect_fracture(image, confidence_threshold):

    if image is None:
        return None, "Please upload an image."

    # PIL image
    image = Image.fromarray(image).convert("RGB")

    # Model input
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert boxes to original image size
    target_sizes = torch.tensor(
        [[image.height, image.width]]
    ).to(device)

    results = processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes
    )[0]

    # PIL → OpenCV
    image_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    predictions = []

    # Draw detections
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

        # Draw bounding box
        cv2.rectangle(
            image_cv,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        text = f"{class_name} {score:.2f}"

        cv2.putText(
            image_cv,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        predictions.append(
            f"{class_name} — {score:.2f}"
        )

    # Convert back to RGB
    output_image = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2RGB
    )

    # Prediction summary
    if predictions:
        summary = "\n".join(predictions)
    else:
        summary = "No detections above the selected confidence threshold."

    return output_image, summary


# ---------------------------------
# Gradio interface
# ---------------------------------

with gr.Blocks(title="Bone Fracture Detection") as demo:

    gr.Markdown(
        """
        # Bone Fracture Detection using DETR

        Upload an X-ray image to run the trained
        Detection Transformer (DETR) model.
        """
    )

    with gr.Row():

        input_image = gr.Image(
            type="numpy",
            label="Upload X-ray"
        )

        output_image = gr.Image(
            type="numpy",
            label="Detection Result"
        )

    confidence = gr.Slider(
        minimum=0.10,
        maximum=0.90,
        value=0.35,
        step=0.05,
        label="Confidence Threshold"
    )

    detect_button = gr.Button(
        "Detect Fracture"
    )

    predictions = gr.Textbox(
        label="Predictions",
        lines=5
    )

    detect_button.click(
        fn=detect_fracture,
        inputs=[
            input_image,
            confidence
        ],
        outputs=[
            output_image,
            predictions
        ]
    )

    gr.Markdown(
        """
        **Disclaimer:** This is an academic/project demonstration
        and is not intended for clinical diagnosis.
        """
    )


# ---------------------------------
# Launch
# ---------------------------------

if __name__ == "__main__":
    demo.launch()