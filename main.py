from argparse import ArgumentParser
import os
from tqdm import tqdm
import os
import re
import cv2
from moviepy.editor import concatenate_videoclips, VideoFileClip
from utils import *


def parse_boolean(value):
    value = value.upper()
    if value == "TRUE":
        return True
    return False


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--reference-imgs-dir', type=str,
                        help='The path of reference images (absolute path required)')
    parser.add_argument('--video', type=str, help='Video that will process')
    parser.add_argument('--output-dir', type=str,
                        help='The path for saving videos')
    parser.add_argument('--face-similarity-threshold', type=float,
                        help='Threshold for minimum face similarity')
    parser.add_argument('--merge-threshold', type=float)
    parser.add_argument('--do-merge', type=parse_boolean, default="False",
                        help='Wheather the clips will be merged or not')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    reference_imgs_dir = args.reference_imgs_dir
    source_video = args.video
    output_dir = args.output_dir
    face_similarity_threshold = args.face_similarity_threshold
    merge_threshold = args.merge_threshold
    do_merge = args.do_merge

    try:
        if not os.path.exists(reference_imgs_dir):
            raise FileNotFoundError(
                f"Error: Reference images directory '{reference_imgs_dir}' not found.")

        os.makedirs(output_dir, exist_ok=True)

        reference_imgs_embeddings = get_reference_faces_embeddings(
            reference_imgs_dir)

        video = cv2.VideoCapture(source_video)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(video.get(cv2.CAP_PROP_FPS))
        print(f'Frame count: {frame_count} | FPS: {fps}')

        print("Processing video")
        pbar = tqdm(ncols=100)
        pbar.reset(total=frame_count)

        frame_intervals = []
        prev_frame_cnt = -1
        curr_frame_cnt = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = get_faces(frame)

            if len(faces) > 0:
                similarity = find_max_similarity(
                    reference_imgs_embeddings, faces)

                if similarity >= face_similarity_threshold:
                    if curr_frame_cnt - prev_frame_cnt > 1:
                        if len(frame_intervals) == 0:
                            frame_intervals.append([curr_frame_cnt, None])
                        else:
                            frame_intervals[-1][-1] = prev_frame_cnt
                            frame_intervals.append([curr_frame_cnt, None])
                        prev_frame_cnt = curr_frame_cnt

            curr_frame_cnt += 1
            pbar.update()

        video.release()

        if len(frame_intervals) and frame_intervals[-1][-1] is None:
            frame_intervals[-1][-1] = prev_frame_cnt

        clip_durations = get_clip_durations(frame_intervals, fps)
        processed_clip_durations = merge_clips(clip_durations, merge_threshold)
        print(f'\nNumber of clips: {len(processed_clip_durations)}')

        print("Start clipping video")
        clip_video(processed_clip_durations, source_video, output_dir)
        print("\nDone Cliping")

        if do_merge:
            print("Merging videos")

            def fn(clipname):
                _, clip_ext = os.path.splitext(clipname)
                regx = f'clip_(\\d+)\\{clip_ext}'
                match = re.search(regx, os.path.basename(clipname))
                clip_number = int(match.group(1))
                return clip_number

            video_filenames = glob(f'{output_dir}/*')
            video_filenames = sorted(video_filenames, key=fn)
            clips = [VideoFileClip(filename) for filename in video_filenames]
            final_clip = concatenate_videoclips(clips)

            _, ext = os.path.splitext(source_video)
            final_clip.write_videofile(
                os.path.join(output_dir, f'merged{ext}'))

    except Exception as e:
        print(f'Error: {e}')
    finally:
        video.release()
