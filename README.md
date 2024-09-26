## An AI-based application designed to separate video clips based on reference faces

### Description
Seperate clips from a video based on reference images. It performs better when several reference faces are given. First, embeddings of reference images are calculating using [**InceptionResnetV1** 
model](https://github.com/timesler/facenet-pytorch.git). Then read each frame from the video and seperate faces from the frame using [**yolo-face**](https://github.com/akanametov/yolo-face).
Upon seperating faces, calculate embeddings of those faces and compare with the embedding of the reference faces. This is how it tracks the video frames where the referece faces are
present.
