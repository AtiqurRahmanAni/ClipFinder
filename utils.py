import os
import torch
import os
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import numpy as np
from PIL import Image
from ultralytics import YOLO
from glob import glob
from torch import Tensor
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from torch import Tensor

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nRunning on device: {device}")

current_file_directory = os.path.dirname(os.path.abspath(__file__))

print("Loding models")
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_detection_model = YOLO(os.path.join(current_file_directory,"yolov8l-face.pt")).to(device)
print("Done loding models")

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

def get_faces(img: np.ndarray) -> list:
    with torch.no_grad():
        results = face_detection_model(img, verbose=False)
    
    faces = []
    boxes = results[0].boxes.xyxy.cpu()

    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped = img[y1:y2, x1:x2, :]
        faces.append(cropped)
    return faces


def get_reference_faces_embeddings(reference_imgs_dir:str) -> Tensor:
    img_filenames = glob(f"{reference_imgs_dir}/*.[pj][pn]g") + glob(f"{reference_imgs_dir}/*.jpeg")
    
    if len(img_filenames) == 0:
        raise FileNotFoundError("No reference images found")
    
    images_tensor = torch.stack([transform(Image.open(filename).convert('RGB')) for filename in img_filenames], dim=0).to(device)

    with torch.no_grad():
        embeddings = embedding_model(images_tensor)
    return embeddings


def find_max_similarity(referece_img_embeddings: Tensor, targets: list) -> float:

    img_tensors = [transform(Image.fromarray(img)) for img in targets]
    img_tensors = torch.stack(img_tensors, dim=0).to(device)

    with torch.no_grad():
        embedding = embedding_model(img_tensors)

    similarity = torch.cosine_similarity(referece_img_embeddings.unsqueeze(1), embedding, dim=-1)
    max_similarity = similarity.max()
    return max_similarity


def get_clip_durations(frame_intervals: list, fps: int) -> list:
    clip_durations = []

    for interval in frame_intervals:
        start_time = interval[0] / fps
        end_time = interval[1] / fps
        clip_durations.append([start_time, end_time])

    return clip_durations

def merge_clips(clip_intervals: list, merge_threshold: float) -> list:
    processed_clip_durations = []

    for duration in clip_intervals:
        if len(processed_clip_durations) == 0:
            processed_clip_durations.append(duration)
        else:
            if duration[0] - processed_clip_durations[-1][-1] <= merge_threshold:
                processed_clip_durations[-1][-1] = duration[1]
            else:
                processed_clip_durations.append(duration)

    processed_clip_durations = [duration for duration in processed_clip_durations if duration[-1] - duration[0] >= 1.0]
    return processed_clip_durations


def clip_video(clip_intervals: list, source_video: str, output_dir: str) -> None:
    _, source_video_ext = os.path.splitext(source_video)

    for i, (start, end) in enumerate(clip_intervals):
        output_video = os.path.join(output_dir, f"clip_{i+1}{source_video_ext}")
        ffmpeg_extract_subclip(filename=source_video, t1=start, t2=end, targetname=output_video)