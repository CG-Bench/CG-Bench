import os
import time
import json
import argparse
import random
import hashlib
import requests
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import traceback
import logging

from utils import *

API_BASE = '' # Your api_base here
API_KEY = '' # Your api_key here

headers = {
    'Authorization': API_KEY,
    'Content-Type': 'application/json',
}

def inference(args, sys_prompt, prompt, image_paths):

    content = []
    messages = []

    if image_paths:
        image_base64_strs = image_paths_to_base64_str(image_paths)
        for image_base64_str in image_base64_strs:
            content.append({'type': 'image_url', 'image_url': {'url': image_base64_str, 'detail': 'low'}})

    content.append({'type': 'text', 'text': sys_prompt + prompt})
    messages.append({'role': 'user', 'content': content})

    json_data = {
        'model': f'{args.model_name}-{args.model_size}',
        'messages': messages,
        'stream': False,
        'temperature': 0.0
    }

    try:
        response = requests.post(API_BASE, headers=headers, json=json_data, timeout=300)

        time.sleep(1)

        try:
            response.raise_for_status()

            result = response.json()

            print(result)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

        if 'choices' not in result or result['choices'] is None:
            print("response fail")
            return None

        return result['choices'][0]['message']['content']

    except (requests.exceptions.ReadTimeout, requests.exceptions.SSLError, requests.exceptions.Timeout) as e:
        print(f"Request failed: {e}")
        return None

def process_single_file(args, json_file):
    """Process a single JSON file with error handling"""
    try:

        anno = load_json(json_file)
        image_paths, frame_indices = load_video_pipeline_args(args, anno)

        prompt = get_prompt(args, anno, frame_indices)

        sys_prompt = SYS[args.task_mode]
        
        response = inference(args, sys_prompt, prompt, image_paths)
        result = post_process(args, anno, response)
        
        if result is not None:
            save_result(args, anno, result, json_file)
            
        return json_file, True
    except Exception as e:
        logging.error(f"Error processing {json_file}: {str(e)}")
        logging.error(traceback.format_exc())
        return json_file, False

def main(args):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='processing.log'
    )
    
    json_files = get_json_files(args)

    print(len(json_files))

    total_files = len(json_files)
    logging.info(f"Found {total_files} files to process")
    print(f"Found {total_files} files to process")


    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:

        future_to_file = {
            executor.submit(process_single_file, args, json_file): json_file 
            for json_file in json_files
        }

        successful = 0
        failed = 0
        
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    _, success = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logging.error(f"Unexpected error processing {json_file}: {str(e)}")
                    logging.error(traceback.format_exc())
                    failed += 1
                pbar.update(1)

    logging.info(f"Processing completed. Successful: {successful}, Failed: {failed}")

def get_args():

    parser = argparse.ArgumentParser(description='run api')

    parser.add_argument('--num_threads', type=int, required=True,
                    help='Number of segments')


    parser.add_argument('--anno_root', type=str, default="./cg_annotations",
                    help='Model name')
    parser.add_argument('--image_root', type=str, default="./cg_images",
                    help='Model name')
    parser.add_argument('--sub_root', type=str, default="./cg_subtitles",
                    help='Model name')

    parser.add_argument('--task_mode', type=str, required=True,
                       choices=['long_acc', 'clue_acc', 'miou', 'open', 
                               'eval_open_step_1', 'eval_open_step_2'],
                       help='Task mode selection')
    
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--model_size', type=str, required=True,
                       help='Model size')
    parser.add_argument('--num_segment', type=int, required=True,
                       help='Number of segments')
    parser.add_argument('--sub', type=str2bool, default=True,
                       help='Sub parameter (true/false)')
    parser.add_argument('--sub_time', type=str2bool, default=True,
                       help='Sub time parameter (true/false)')
    parser.add_argument('--frame_time', type=str2bool, default=True,
                       help='Frame time parameter (true/false)')
    
    parser.add_argument('--open_model_name', type=str,
                       help='Open model name')
    parser.add_argument('--open_model_size', type=str,
                       help='Open model size')
    parser.add_argument('--open_num_segment', type=int,
                       help='Open number of segments')
    parser.add_argument('--open_sub', type=str2bool, default=True,
                       help='Open sub parameter (true/false)')
    parser.add_argument('--open_sub_time', type=str2bool, default=True,
                       help='Open sub time parameter (true/false)')
    parser.add_argument('--open_frame_time', type=str2bool, default=True,
                       help='Open frame time parameter (true/false)')
    
    args = parser.parse_args()
    

    if args.task_mode in ['eval_open_step_1', 'eval_open_step_2']:
        if not all([args.open_model_name, args.open_model_size, args.open_num_segment]):
            parser.error('eval_open_step_1 and eval_open_step_2 require open_model_name, '
                        'open_model_size, and open_num_segment')

    with open("./video_meta_info.json", "r") as f:
        args.vdict = json.load(f)

    if args.task_mode == "clue_acc":
        if args.num_segment > 32:
            args.num_segment = 32

    if args.task_mode == "eval_open_step_1":
        args.sub = False
        args.frame_time = False

    if args.task_mode == "eval_open_step_2":
        args.sub = True
        args.frame_time = True        

    if not args.sub:
        args.sub_time = False

    return args
    

if __name__ == "__main__":
    args = get_args()
    main(args)