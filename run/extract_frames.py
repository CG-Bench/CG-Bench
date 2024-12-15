import os
import numpy as np
import json
import argparse
import decord
from PIL import Image
import concurrent.futures
from tqdm import tqdm


def process_video_frame(video_uid, frame_indices, cg_videos_path, output_images_path):
    video_path = os.path.join(cg_videos_path, f"{video_uid}.mp4")
    video_output_path = os.path.join(output_images_path, video_uid)
    os.makedirs(video_output_path, exist_ok=True)


    vr = decord.VideoReader(video_path)


    for frame_idx in frame_indices:

        frame_filename = os.path.join(video_output_path, f"{frame_idx}.jpg")
        if os.path.exists(frame_filename):
            continue

        # 抽帧并保存
        try:
            frame = vr[frame_idx].asnumpy() 
            image = Image.fromarray(frame)  
            image.save(frame_filename, 'JPEG') 
        except IndexError:
            print(f"frame {frame_idx} out of range, skip")

def process_global_frames(cg_videos_path, video_meta_info, output_images_path, num_segment):
    video_uids = list(set(os.path.splitext(f)[0] for f in os.listdir(cg_videos_path) if f.endswith('.mp4'))) 
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for video_uid in video_uids:
            video_path = os.path.join(cg_videos_path, f"{video_uid}.mp4")
            max_frame = video_meta_info[video_uid]["max_frame"] 
            frame_indices = sample_frames_global_average(max_frame, num_segment)

            futures.append(executor.submit(process_video_frame, video_uid, frame_indices, cg_videos_path, output_images_path))

        for _ in tqdm(concurrent.futures.as_completed(futures), desc="process", total=len(futures), ncols=100):
            pass

def process_cgbench_data(cgbench_data, video_meta_info, cg_videos_path, output_images_path, num_segment):

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for item in cgbench_data:
            video_uid = item['video_uid']
            clue_intervals = item.get('clue_intervals', [])

            if video_uid in video_meta_info:
                video_info = video_meta_info[video_uid]
                fps = video_info['fps']
                max_frame = video_info['max_frame']
                frame_indices = sample_frames_clue_average(clue_intervals, num_segment, fps)

                futures.append(executor.submit(process_video_frame, video_uid, frame_indices, cg_videos_path, output_images_path))

        for _ in tqdm(concurrent.futures.as_completed(futures), desc="process", total=len(futures), ncols=100):
            pass

def sample_frames_global_average(max_frame, num_segment):
    frame_indices = []
    if num_segment != 0.0:
        seg_size = float(max_frame) / num_segment
        frame_indices = np.array([ 
            int(seg_size / 2 + np.round(seg_size * idx)) 
            for idx in range(num_segment)
        ])
    return frame_indices

def sample_frames_clue_average(clue_intervals, num_segment, fps):
    clues_frame_intervals = [(round(interval[0] * fps), round(interval[1] * fps)) for interval in clue_intervals]
    clue_durations = [interval[1] - interval[0] for interval in clues_frame_intervals]
    total_duration = sum(clue_durations)
    if num_segment >= total_duration:
        return [frame for interval in clues_frame_intervals for frame in range(interval[0], interval[1])]
    frames_per_clue = [int(num_segment * (duration / total_duration)) for duration in clue_durations]
    frame_indices = []
    for i, (interval, num_frames) in enumerate(zip(clues_frame_intervals, frames_per_clue)):
        num_frames = max(1, num_frames)
        seg_size = (interval[1] - interval[0]) / num_frames
        clue_frame_indices = [
            int(interval[0] + seg_size / 2 + seg_size * idx) for idx in range(num_frames)
        ]
        frame_indices.extend(clue_frame_indices)
    return frame_indices

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--method', choices=['global', 'interval'], required=True, help="")
    parser.add_argument('--num_segment', type=int, required=True, help="")
    return parser.parse_args()

def main():

    args = parse_args()

    video_meta_info_path = './video_meta_info.json'
    cgbench_json_path = './cgbench_mini.json'
    cg_videos_path = './cg_videos_720p/'
    output_images_path = './cg_images/'

    with open(video_meta_info_path, 'r', encoding='utf-8') as f:
        video_meta_info = json.load(f)

    with open(cgbench_json_path, 'r', encoding='utf-8') as f:
        cgbench_data = json.load(f)

    if args.method == 'global':
        process_global_frames(cg_videos_path, video_meta_info, output_images_path, num_segment=args.num_segment)
    
    elif args.method == 'interval':
        process_cgbench_data(cgbench_data, video_meta_info, cg_videos_path, output_images_path, num_segment=args.num_segment)
    
    print("complete")

if __name__ == '__main__':
    main()
