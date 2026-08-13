# Custom 5-Class Vision AI on NVIDIA Jetson

This project is a real-time, custom 5-class image classification AI built for the NVIDIA Jetson edge computing platform. 
The program captures live video from a USB/CSI camera or reads static images, processes them through a custom-trained ResNet-18 model, and instantly displays the classification results and confidence scores on the screen.

## Demo Video
This video demonstrates the complete real-time classification pipeline running on the NVIDIA Jetson.

[Watch the demonstration video here - https://drive.google.com/file/d/1apZxJLKIYde8vCKuYIOPc5vA5mpU6Bj1/view?usp=drive_link]

## Project Screenshots

### 1. Live Camera Detection
The AI successfully classifies objects in real-time through a USB camera feed, displaying the predicted class and confidence percentage.

### 2. Static Image Batch Processing
The program can also process static image files, overlaying the AI's judgment directly onto the generated output images.

### 3. Complete Hardware Setup
The complete prototype uses an NVIDIA Jetson device and a USB/CSI camera connected to the development board.

## Why This Project Matters
Modern computer vision systems often rely on cloud servers, which introduces network latency and privacy concerns. However, in fields like robotics, automated manufacturing, and smart agriculture, systems must make instant decisions on the spot.

This project demonstrates the power of **Edge AI**. By training a custom deep learning model (ResNet-18) and optimizing it with NVIDIA TensorRT, the AI runs entirely locally on the Jetson device. It proves that we can build highly accurate, domain-specific AI models (e.g., recognizing specific animal species, mechanical parts, or hand gestures) that operate completely offline with zero cloud latency.

## Features
* Captures real-time video using a standard USB or CSI camera
* Processes static image files for quick evaluation
* Classifies objects into 5 custom categories
* Powered by a custom-trained PyTorch ResNet-18 backbone
* Highly optimized for Jetson using TensorRT (ONNX format)
* Displays prediction results and confidence scores visually
* Runs the complete inference pipeline with a single command

## How It Works
1. The user provides an input (USB camera feed or static `.jpg` image).
2. The image is scaled and normalized to $224 \times 224$ pixels.
3. The Jetson processes the image through the custom ONNX model (`resnet18.onnx`).
4. TensorRT accelerates the inference process on the Jetson's GPU.
5. A Softmax layer calculates the probability score for each of the 5 classes.
6. The Top-1 result (highest probability) is displayed as an overlay on the output image/video.

## Technologies
* NVIDIA Jetson platform
* Python 3
* PyTorch
* NVIDIA TensorRT
* ONNX (Open Neural Network Exchange)
* `jetson-inference` framework
* Docker

## Project Structure
```text
NVIDIA13/
├── python/
│   └── training/
│       └── classification/
│           ├── models/
│           │   └── Final/
│           │       ├── resnet18.onnx   # The trained and optimized AI model
│           │       └── labels.txt      # The 5 custom class labels
│           ├── data/                   # (Local only) Training and test image datasets
│           └── test/                   # (Local only) Images used for evaluation
├── README.md
└── ...

```

## Tested Environment

This project was tested with:

* NVIDIA Jetson platform
* JetPack OS (Linux for Tegra)
* `jetson-inference` Docker container
* Standard USB V4L2 camera

*Other environments may require different configurations or package versions.*

## Installation

### 1. Launch the Docker Container

To avoid dependency issues, this project uses the official `jetson-inference` container.

```bash
cd ~/jetson-inference
./docker/run.sh

```

*Note: The following commands should be executed inside the container.*

### 2. Navigate to the Project Directory

```bash
cd /jetson-inference/python/training/classification

```

## Run the Program

### Option A: Real-Time Camera Inference

To run the AI using a connected USB camera (`/dev/video0`):

```bash
imagenet --model=models/Final/resnet18.onnx \
         --labels=models/Final/labels.txt \
         --input_blob=input_0 \
         --output_blob=output_0 \
         /dev/video0

```

### Option B: Static Image Inference

To evaluate a specific saved image (e.g., `test/Cat.jpg`) and generate a result image (`result_cat.jpg`):

```bash
imagenet --model=models/Final/resnet18.onnx \
         --labels=models/Final/labels.txt \
         --input_blob=input_0 \
         --output_blob=output_0 \
         test/Cat.jpg \
         result_cat.jpg

```

## Output

When running static images, the program will generate a new image file (e.g., `result_cat.jpg`) in your directory. If you open this file, you will see the original image with text overlaid at the top, such as:
`98.5% class_name`

## Important Notes & Limitations

* **Confidence is not absolute:** The AI outputs a probability score based on its training data. A 99% score means the AI is highly confident based on what it has learned, but it can still make mistakes on unseen data.
* **Lighting and Angles:** Results heavily depend on the lighting, camera angle, and background of the target object.
* **Dataset Bias:** The model can only recognize the 5 specific classes it was trained on. If you show it an object outside these 5 classes, it will incorrectly force it into one of the known categories.
* Datasets (`.jpg` files) and raw PyTorch checkpoints (`.pth.tar`) are excluded from this repository to save space; only the final ONNX model is provided.

```

```
