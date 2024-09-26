## An AI-based application designed to separate video clips based on reference faces

### Description
This application separates clips from a video based on reference images. It performs optimally when multiple reference faces are provided. First, embeddings of the reference images are calculated using the [**InceptionResnetV1** 
model](https://github.com/timesler/facenet-pytorch.git). Then, each frame from the video is processed, and faces are detected using [**yolo-face**](https://github.com/akanametov/yolo-face). After face detection, embeddings of the detected faces are calculated and compared with the embeddings of the reference faces to track video frames where the reference faces appear.

### Project requirements
on version **3.10** or higher is required to run the backend of this project. It may work on earlier versions as well. A minimum of **6GB GPU VRAM** is recommended for faster processing.

### How to run
- Navigate to the project root directory.
- Run `python -m venv venv` and `.\venv\Scripts\activate` to create and activate a virtual environment (for Windows OS).
- Run `pip install -r requirements.txt` to install all dependencies.
- Download the [yolov8l-face.pt](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt) and place it in the project root directory.
- Copy the command from [**runcommand**](https://github.com/AtiqurRahmanAni/ClipFinder/blob/main/runcommand.txt), modify it according to your folder structure, and run the command to start processing.

**Note:** Install `PyTorch` based on your device configuration by referring to the [installation page](https://pytorch.org/get-started/locally/). If you are using **NVIDIA GPU**, 
you do not need to install `torch==2.4.1+rocm6.1`, `torchaudio==2.4.1+rocm6.1`, and `torchvision==0.19.1+rocm6.1`. Remove them from the `requirements.txt` file before executing `pip install -r requirements.txt`. This also applies if you don't have a dedicated GPU.

### Project demo
YouTube: [Link](https://youtu.be/cDSp6RYn1lY?si=8EnwzjbIKAyrsVHy)
