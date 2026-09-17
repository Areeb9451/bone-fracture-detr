\# Bone Fracture Detection using DETR



An object detection project for detecting and localizing bone-fracture-related patterns in X-ray images using a fine-tuned \*\*Detection Transformer (DETR) with a ResNet-50 backbone\*\*.



The trained model is deployed as an interactive Gradio web application using Hugging Face Spaces and ZeroGPU.



\## 🚀 Live Demo



\*\*Hugging Face Space:\*\*

https://huggingface.co/spaces/Areeb9451/bone-fracture-detr-demo



\*\*Hugging Face Model:\*\*

https://huggingface.co/Areeb9451/bone-fracture-detr



\## 📌 Project Overview



This project explores transformer-based object detection for identifying fracture-related regions in X-ray images.



The model was developed using transfer learning from the pretrained:



`facebook/detr-resnet-50`



The model was fine-tuned on a bone-fracture X-ray dataset and achieved an approximately \*\*55 mAP\*\* during evaluation.



The final model predicts bounding boxes around detected regions and assigns one of five classes.



\## 🧠 Model Architecture



\*\*Detection Transformer (DETR)\*\*



\* Backbone: ResNet-50

\* Framework: PyTorch

\* Model library: Hugging Face Transformers

\* Pretrained checkpoint: `facebook/detr-resnet-50`

\* Training approach: Transfer learning / fine-tuning

\* Output: Object bounding boxes + class labels + confidence scores



DETR treats object detection as a direct set-prediction problem and uses a Transformer-based architecture instead of a traditional region proposal pipeline.



\## 🏷️ Detection Classes



The model uses the following class mapping:



| ID | Class           |

| -: | --------------- |

|  0 | bone-fracture   |

|  1 | angle           |

|  2 | fracture        |

|  3 | line            |

|  4 | messed\_up\_angle |



\## 📊 Performance



The trained model achieved approximately:



\*\*mAP: \~55\*\*



The reported performance is based on the evaluation performed during the original project training.



Because the dataset used for training was relatively limited, the model may have limited generalization to X-ray images from different sources, imaging systems, patient populations, or acquisition conditions.



\## 🔬 Inference Pipeline



The application follows this workflow:



```text

X-ray Image

&#x20;    ↓

Image Preprocessing

&#x20;    ↓

DETR + ResNet-50

&#x20;    ↓

Object Detection

&#x20;    ↓

Bounding Box Post-processing

&#x20;    ↓

Confidence Filtering

&#x20;    ↓

Annotated X-ray + Predictions

```



\## 🖥️ Web Application



The project includes a Gradio interface that allows users to:



1\. Upload an X-ray image.

2\. Select a confidence threshold.

3\. Run DETR inference.

4\. View the detected bounding boxes.

5\. View predicted classes and confidence scores.



The live application is hosted on Hugging Face Spaces using ZeroGPU.



\## 📁 Project Structure



```text

bone-fracture-detr/

│

├── app.py

├── detect.py

├── inference.py

├── requirements.txt

├── README.md

├── .gitignore

│

└── model/

&#x20;   ├── config.json

&#x20;   └── preprocessor\_config.json

```



The trained `model.safetensors` file is not stored in this GitHub repository because of GitHub's file-size limitations. The trained model is hosted separately on Hugging Face.



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/Areeb9451/bone-fracture-detr.git

cd bone-fracture-detr

```



Create a virtual environment:



\### Windows



```powershell

python -m venv venv

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## 🤗 Model



The trained model is available on Hugging Face:



https://huggingface.co/Areeb9451/bone-fracture-detr



The application automatically loads the model from the Hugging Face Hub when using the current `app.py` implementation.



\## ▶️ Run Locally



Start the Gradio application:



```bash

python app.py

```



The application will start a local Gradio server.



Open the local URL shown in the terminal and upload an X-ray image for inference.



\## 🔍 Command-Line Inference



The repository also contains inference utilities.



For example:



```bash

python inference.py

```



The detection script can be used to process an image and generate an annotated result:



```bash

python detect.py <image\_path>

```



\## 🛠️ Technologies Used



\* Python

\* PyTorch

\* Torchvision

\* Hugging Face Transformers

\* DETR

\* ResNet-50

\* OpenCV

\* NumPy

\* Pillow

\* Gradio

\* Hugging Face Hub

\* Hugging Face Spaces



\## 📚 Learning Outcomes



This project provided practical experience with:



\* Transformer-based object detection

\* DETR architecture

\* Transfer learning

\* Object detection datasets and annotations

\* Bounding-box prediction

\* Model inference and post-processing

\* PyTorch model deployment

\* Gradio web interfaces

\* Hugging Face model hosting

\* Hugging Face Spaces

\* ZeroGPU deployment

\* Git and GitHub project management



\## ⚠️ Limitations



\* The training dataset was relatively limited.

\* Performance may vary on images from different sources.

\* The reported mAP should not be interpreted as clinical validation.

\* The model has not been validated for use in a real



