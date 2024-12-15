import io
import os
import re
import json
import base64
import pysubs2
import os.path as osp

import numpy as np

from PIL import Image

SYS = {

    'long_acc': (
        "You will be provided with sampled frames from a video, along with a multiple-choice question that includes a question and several answer options.\n"
        "Your task is to analyze the provided frames, infer the most plausible answer based on the visual information.\n"
        "If the video does not provide enough information, infer the answer based on the options available and still provide a result. "
        "Therefore, In all cases, an answer must be given.\n"
        "Only output the answer in the following format:\n\n"
        "```json\n{\"result\": \"option\"}\n```\n\n"
        "The \"option\" is the uppercase letter corresponding to your answer.\n\n"
    ),

    'clue_acc': (
        "You will be provided with sampled frames from a video, along with a multiple-choice question that includes a question and several answer options.\n"
        "Your task is to analyze the provided frames, infer the most plausible answer based on the visual information.\n"
        "If the video does not provide enough information, infer the answer based on the options available and still provide a result. "
        "Therefore, In all cases, an answer must be given.\n"
        "Only output the answer in the following format:\n\n"
        "```json\n{\"result\": \"option\"}\n```\n\n"
        "The 'option' is the uppercase letter corresponding to your answer.\n\n"
    ),

    'miou': (
        "You will be provided with uniformly sampled frames from a video and their timestamps, along with a multiple-choice question that includes a question and several answer options.\n"
        "Your task is to determine in which intervals the 'clue intervals' exist that contain visual information needed to answer the question.\n"
        "Only output the answer in the following format:\n\n"
        "```json\n{\"result\": [[start1, end1], [start2, end2], ...]}\n```\n\n"
        "In this output format, each 'start' and 'end' represents the beginning and end of an interval in seconds where relevant clues can be found.\n"
        "You must provide at least one interval and at most five intervals. Intervals exceeding five will NOT be considered valid.\n"
    ),

    'open': (
        "You will be provided with sampled frames from a video, along with a question.\n"
        "Your task is to analyze the provided frames and infer the most plausible answer based on the visual information.\n"
        "If the visual information is ambiguous or insufficient, use the available context to reason your answer.\n"
        "Only output the answer in the following format:\n\n"
        "```json\n{\"result\": \"answer\"}\n```\n\n"
        "The \"answer\" can be a word, phrase, or sentence that directly responds to the question.\n\n"
    ),

    'eval_open_step_1': (
        "You will be provided with a question, a model's prediction, and the ground truth answer for this question.\n"
        "Your task is to judge whether the model's prediction is correct based on the meaning of the two texts.\n"
        "In most cases, this can be done by determining if the meaning of the model's prediction is consistent with, or contains, the ground truth answer. However, in some cases where the two texts differ, it may represent different descriptions of the same visual scene, in which case visual information is needed for further judgment.\n"
        "Therefore, I hope you:\n"
        "- Output 0, if the model's prediction and the ground truth answer are neither consistent nor related by inclusion, with fundamentally different meanings.\n"
        "- Output 1, if the meaning of the model's prediction and the ground truth answer is consistent, or if the model's prediction meaningfully contains the ground truth answer.\n"
        "- Output 2, if the model's prediction and ground truth are not consistent or inclusive, but may be different descriptions of the same visual scene, requiring visual information for further judgment.\n"
        "Only output the answer in the following format:\n\n"
        "```json\n{\"result\": choice}\n```\n\n"
        "The choice is either 0, 1, or 2 as specified above."
    ),

    'eval_open_step_2': (
        "You will be provided with a question, a model's prediction, and the sampling frames of the clue intervals related to this question.\n"
        "Your task is to determine whether the model has answered the question correctly based on the visual information provided.\n"
        "Therefore, I hope you:\n"
        "- Output 0, if the model's prediction does not correctly answer the question.\n"
        "- Output 1, if the model's prediction correctly answers the question.\n"
        "Only output the answer in the following format without extra explanation:\n\n"
        "```json\n{\"result\": choice}\n```\n\n"
        "The choice is either 0 or 1 as specified above."
    )

}

def load_json(json_file):

    with open(json_file, "r") as f:
        anno = json.load(f)

    return anno

def save_json(anno, json_file):

    with open(json_file, "w") as f:
        
        json.dump(anno, f, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list.
    Assumes each interval is a list [start, end].
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # Check if there is an overlap
        if current[0] <= last_merged[1]:
            # Merge the current interval with the last one
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # No overlap, add current interval
            merged.append(current)
    
    return merged

def calculate_intervals_iou(intervals1, intervals2):
    """
    Calculate the IoU of two lists of intervals.
    Each list contains intervals represented as [start, end].
    """
    # Merge overlapping intervals in both lists
    merged1 = merge_intervals(intervals1)
    merged2 = merge_intervals(intervals2)
    
    # Calculate total length of intervals for both lists
    def total_length(merged_intervals):
        return sum(end - start for start, end in merged_intervals)
    
    length1 = total_length(merged1)
    length2 = total_length(merged2)
    
    # Calculate intersection length
    intersection_length = 0
    for interval1 in merged1:
        for interval2 in merged2:
            intersection_start = max(interval1[0], interval2[0])
            intersection_end = min(interval1[1], interval2[1])
            intersection_length += max(0, intersection_end - intersection_start)
    # Calculate union length
    union_length = length1 + length2 - intersection_length
    # IoU is intersection divided by union
    iou = intersection_length / union_length if union_length > 0 else 0
    return iou

def milliseconds_to_seconds(milliseconds):
    return milliseconds / 1000

def image_paths_to_base64_str(image_paths):

    image_base64_strs = []

    for image in image_paths:
        with Image.open(image) as img:
            byte_stream = io.BytesIO()
            img.save(byte_stream, format='jpeg')
            encoded_string = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
        image_base64_strs.append(f'data:image/jpeg;base64,{encoded_string}')

    return image_base64_strs

def get_list_image_paths(image_dir, frame_indices):
    valid_image_paths = []
    valid_frame_indices = []
    for frame_index in frame_indices:
        image_path = osp.join(image_dir, f"{frame_index}.jpg")
        if osp.exists(image_path):
            valid_image_paths.append(image_path)
            valid_frame_indices.append(frame_index)
    return valid_image_paths, valid_frame_indices

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

def load_video_pipeline_args(args, anno):

    if args.task_mode in ["long_acc", "miou", "open"]:
        return load_video_pipeline(osp.join(args.image_root, anno["video_uid"]), args.vdict[anno["video_uid"]]["max_frame"], args.vdict[anno["video_uid"]]["fps"], args.num_segment)
    elif args.task_mode in ["clue_acc", "eval_open_step_2"]:
        return load_video_pipeline(osp.join(args.image_root, anno["video_uid"]), None, args.vdict[anno["video_uid"]]["fps"], args.num_segment, anno["clue_intervals"])
    elif args.task_mode == "eval_open_step_1":
        return [], []

def load_video_pipeline(image_dir, max_frame, fps, num_segment, clue_intervals=None):

    if clue_intervals:
        frame_indices = sample_frames_clue_average(clue_intervals, num_segment, fps)
    else:
        frame_indices = sample_frames_global_average(max_frame, num_segment)

    # print(image_dir)

    # print(frame_indices)

    image_paths, frame_indices = get_list_image_paths(image_dir, frame_indices)

    # print(image_paths, frame_indices)

    # exit()

    return image_paths, frame_indices

def get_subtitles(sub_dir, video_uid, fps, frame_indices, sub_time=False):

    subtitles = []

    srt_path = osp.join(sub_dir, f"{video_uid}.srt")
    if osp.exists(srt_path):
        subs = pysubs2.load(srt_path, encoding="utf-8")
        for frame_index in frame_indices:
            cur_time = pysubs2.make_time(fps=fps, frames=frame_index)
            for sub in subs:
                if sub.start < cur_time and sub.end > cur_time:
                    sub_text = sub.text.replace("\\N", " ")
                    if sub_time:
                        start_time = milliseconds_to_seconds(sub.start)
                        end_time = milliseconds_to_seconds(sub.end)
                        sub_text = f"[{start_time}, {end_time}] {sub_text}"
                    if sub_text.strip() and sub_text not in subtitles:
                        subtitles.append(sub_text)

    if subtitles:
        subtitles_str = '\n'.join(subtitles)
        return f"The subtitles of the video are as follows:\n\n{subtitles_str}\n\n"
    else:
        return ""

def get_frame_times(fps, frame_indices):
    seconds = list(map(lambda x: str(round(x / fps, 4)), frame_indices))
    timestamps = ", ".join(seconds)
    return f"A total of {len(frame_indices)} frames are sampled. Their corresponding timestamps are:\n\n{timestamps}\n\n"

def get_json_files(args):
    json_files = []
    
    print(args.anno_root)

    for root, _, files in os.walk(args.anno_root):
        for file in files:
            if not file.endswith('.json'):
                continue
                
            json_path = os.path.join(root, file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    anno = json.load(f)
                
                result_key = f"{args.task_mode}_{args.model_name}_{args.model_size}_{args.num_segment}_{args.sub}_{args.sub_time}_{args.frame_time}"
                
                # print(result_key)

                # exit()

                needs_evaluation = False
                if args.task_mode in ['long_acc', 'clue_acc', 'miou', 'open']:
                    if result_key not in anno["results"]:
                        needs_evaluation = True
                    elif anno["results"][result_key]["version"] != anno["version"]:
                        needs_evaluation = True
                
                # eval_open_step_1
                elif args.task_mode == 'eval_open_step_1':
                    open_key = f"open_{args.open_model_name}_{args.open_model_size}_{args.open_num_segment}_{args.open_sub}_{args.open_sub_time}_{args.open_frame_time}"
                    
                    if open_key in anno["results"]:
                        if "step_1" not in anno["results"][open_key]:
                            needs_evaluation = True
                        elif anno["results"][open_key]["step_1"]["version"] != anno["results"][open_key]["version"]:
                            needs_evaluation = True
                
                # eval_open_step_2
                elif args.task_mode == 'eval_open_step_2':
                    open_key = f"open_{args.open_model_name}_{args.open_model_size}_{args.open_num_segment}_{args.open_sub}_{args.open_sub_time}_{args.open_frame_time}"
                    
                    if open_key in anno["results"] and "step_1" in anno["results"][open_key]:
                        if "step_2" not in anno["results"][open_key]:
                            needs_evaluation = True
                        elif anno["results"][open_key]["step_2"]["version"] != anno["results"][open_key]["step_1"]["version"]:
                            needs_evaluation = True
                
                if needs_evaluation:
                    json_files.append(json_path)
                    
            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")
                continue
    
    return json_files

def get_prompt(args, anno, frame_indices):
    
    prompt = ""

    if args.sub:

        prompt += get_subtitles(args.sub_root, anno["video_uid"], args.vdict[anno["video_uid"]]["fps"], frame_indices, args.sub_time)

    if args.frame_time:

        prompt += get_frame_times(args.vdict[anno["video_uid"]]["fps"], frame_indices)

    prompt += f"Question: {anno['question']}\n\n"

    if args.task_mode in ["long_acc", "clue_acc", "miou"]:

        choices = anno['choices']
        labels = [chr(ord('A') + i) for i in range(len(choices))]
        prompt += "\n".join([f"{label}:{value}" for label, value in zip(labels, choices)]) + "\n\n"

    if args.task_mode == "eval_open_step_1":

        prompt += f"The ground truth answer is \'{anno['answer']}\'\n\n"

    if args.task_mode in ["eval_open_step_1", "eval_open_step_2"]:

        open_key = f"open_{args.open_model_name}_{args.open_model_size}_{args.open_num_segment}_{args.open_sub}_{args.open_sub_time}_{args.open_frame_time}"
        prompt += f"The model's prediction is \'{anno['results'][open_key]['result']}\'\n\n"

    return prompt    

def save_result(args, anno, result, json_file):

    if args.task_mode in ["long_acc", "clue_acc", "miou", "open"]:    
        result_key = f"{args.task_mode}_{args.model_name}_{args.model_size}_{args.num_segment}_{args.sub}_{args.sub_time}_{args.frame_time}"
        if result_key not in anno["results"]:
            anno["results"][result_key] = {}
        anno["results"][result_key].update({
            "version": anno["version"],
            "result": result
        })

        save_json(anno, json_file)

    elif args.task_mode in ["eval_open_step_1"]:
        open_key = f"open_{args.open_model_name}_{args.open_model_size}_{args.open_num_segment}_{args.open_sub}_{args.open_sub_time}_{args.open_frame_time}"
        if "step_1" not in anno["results"][open_key]:
            anno["results"][open_key]["step_1"] = {}
        anno["results"][open_key]["step_1"].update({
            "version": anno["results"][open_key]["version"],
            "result": result
        })
        if result in [0, 1]:            
            if "step_2" not in anno["results"][open_key]:
                anno["results"][open_key]["step_2"] = {}
            anno["results"][open_key]["step_2"].update({
                "version": anno["results"][open_key]["version"],
                "result": result
            })

        save_json(anno, json_file)

    elif args.task_mode in ["eval_open_step_2"]:
        open_key = f"open_{args.open_model_name}_{args.open_model_size}_{args.open_num_segment}_{args.open_sub}_{args.open_sub_time}_{args.open_frame_time}"
        if "step_2" not in anno["results"][open_key]:
            anno["results"][open_key]["step_2"] = {}
        anno["results"][open_key]["step_2"].update({
            "version": anno["results"][open_key]["step_1"]["version"],
            "result": result
        })

        save_json(anno, json_file)

def post_process(args, anno, response):

    result = None

    if response:

        json_start = response.find('```json')
        json_end = response.find('```', json_start + len('```json'))
        if json_start != -1 and json_end != -1:
            json_content = response[json_start + len('```json'):json_end].strip()
        else:
            json_content = ""
        if json_content:
            if args.task_mode in ["long_acc", "clue_acc"]:
                json_content = re.sub(r'(?<=:\s)([A-Za-z_]\w*)', r'"\1"', json_content)
            try:
                model_result = json.loads(json_content)["result"]
                if args.task_mode in ["long_acc", "clue_acc"]:
                    right_answer = anno["right_answer"]
                    result = 1 if right_answer == model_result else 0
                elif args.task_mode == "miou":
                    right_answer = anno["clue_intervals"]
                    result = calculate_intervals_iou(model_result, right_answer)
                elif args.task_mode in ["open", "eval_open_step_1", "eval_open_step_2"]:
                    result = model_result
            except Exception as e:
                print(f"Error in parsing JSON: {e}, {json_content}")
        if result == None:
            if args.task_mode in ["long_acc", "clue_acc"]:
                matches = re.findall(r'\b[A-H]\b', response)
                if matches:
                    right_answer = anno["right_answer"]
                    result = 1 if right_answer in matches else 0
            elif args.task_mode == "miou":
                numbers = re.findall(r'-?\d+\.?\d*', response)
                if len(numbers) % 2 != 0:
                    result = None
                else:
                    right_answer = anno["clue_intervals"]
                    model_result = [(float(numbers[i]), float(numbers[i+1])) for i in range(0, len(numbers), 2)]
                    result = calculate_intervals_iou(model_result, right_answer)
            elif args.task_mode == "eval_open_step_1":
                match = re.search(r'[012]', response)
                if match:
                    result = int(match.group())
            elif args.task_mode == "eval_open_step_2":
                match = re.search(r'[01]', response)
                if match:
                    result = int(match.group())
            elif args.task_mode == "open":
                result = response
        
    return result
