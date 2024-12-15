import json
import os

from tqdm import tqdm

input_file_path = "./cgbench_mini.json"

output_dir = "./cg_annotations/"

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

keys_to_save = ['qid', 'video_uid', 'question', 'answer', 'choices', 'right_answer', 'clue_intervals']

changed_data = 0
new_data = 0

for item in tqdm(data, desc="Processing jsons", unit="file"):

    filtered_item = {key: item[key] for key in keys_to_save}

    qid = filtered_item['qid']
    output_file_path = os.path.join(output_dir, f"{qid}.json")

    if os.path.exists(output_file_path):

        with open(output_file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        changes = False
        for key in ['question', 'answer', 'choices', 'right_answer', 'clue_intervals']:
            if existing_data[key] != filtered_item[key]:
                changes = True
                break

        if changes:
            changed_data += 1
            filtered_item['version'] = existing_data['version'] + 1
            filtered_item["results"] = existing_data["results"]
        else:
            continue
    else:
        filtered_item['version'] = 0
        filtered_item['results'] = {}
        new_data += 1

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_item, f, ensure_ascii=False, indent=4)


print(f"changed_data: {changed_data}, new_data: {new_data}")