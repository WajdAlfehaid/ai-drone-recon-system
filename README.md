# Real-Time Semantic Segmentation & Threat Monitoring System
**AI-Powered Drone HUD**

**Author:** Wajd Alfehaid  
**Script Name:** `drone_recon.py`  
**Tech Stack:** Python, PyTorch, OpenCV, Hugging Face Transformers

---

## 👁️ System Overview
The **Sentinel System** is a computer vision application designed for autonomous drones or stationary security cameras. It utilizes **Semantic Segmentation**—a deep learning technique that classifies every single pixel in an image—to understand its environment in real-time.

Unlike standard object detection (which just puts boxes around things), this system understands the *shape* and *context* of the terrain. It separates the world into **Safe Zones** (Floor, Road, Grass) and **Threat/Obstacle Zones** (Walls, Humans, Cars), providing navigation commands to a pilot or flight controller.



---

## ⚙️ Key Features

### 1. AI-Powered Terrain Analysis
The system uses a Transformer-based model (via Hugging Face) to segment the video feed into 150+ potential categories.
* **Safe Classes:** Floor, Grass, Road (Green tint).
* **Threat Classes:** Walls, Humans, Cars, Obstacles (Red tint).

### 2. Heads-Up Display (HUD)
A military-style overlay provides real-time telemetry to the operator:
* **Target Locking:** Automatically draws bounding brackets around the largest identified threat.
* **Range Estimation:** Uses optical geometry to estimate distance (`RNG`) based on target height.
* **Flight Director:** Calculates steering commands (`<< LEFT`, `RIGHT >>`, `LOCKED`) to keep the drone centered on the target or avoid obstacles.



### 3. Mission Recording & Intel
* **Black Box Recording:** Automatically saves the entire mission feed (with HUD) to an `.avi` video file.
* **Intel Snapshots:** Press `S` to capture high-resolution still images of the current frame for analysis.

---

## 🛠️ System Architecture

The pipeline runs in a continuous loop at approximately 20-30 FPS (depending on hardware):

1.  **Sensor Input:** Captures raw frame from USB or CSI camera (`cv2.VideoCapture`).
2.  **Preprocessing:** Resizes frame to 640x480 and normalizes colors for the AI.
3.  **Inference:** The **SegFormer** (or similar) model predicts the class of every pixel.
4.  **Mask Generation:**
    * A **Boolean Mask** filters pixels into "Safe" or "Threat" layers.
    * `cv2.findContours` identifies specific distinct objects within the threat mask.
5.  **Telemetry Calculation:**
    * Calculates the centroid $(C_x, C_y)$ of the largest threat.
    * Computes deviation from the image center $(320, 240)$ to generate steering error.
6.  **Rendering:** Merges the original frame, segmentation color masks, and vector graphics (HUD lines) into the final output.

---

## 💻 Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ and a CUDA-capable GPU (NVIDIA) or Apple Silicon (M1/M2/M3).

### Dependencies
Install the required libraries:
```bash
pip install torch torchvision numpy opencv-python transformers
