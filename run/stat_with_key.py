import os
import json
from tqdm import tqdm  # 导入 tqdm

anno_root = "./cg_annotations"

result_keys = [
    "clue_acc_gpt-4o_2024-08-06_32_True_True_True",
]

# 初始化 result_dict
result_dict = {}
for result_key in result_keys:
    result_dict[result_key] = {
        "total": 0,
        "result": 0,
        "result_total": 0,
        "th_0.1": 0,
        "th_0.2": 0,
        "th_0.3": 0,
        "th_0.4": 0,
        "th_0.5": 0,
        "iou_th_0.1": 0,
        "iou_th_0.2": 0,
        "iou_th_0.3": 0,
        "iou_th_0.4": 0,
        "iou_th_0.5": 0,
    }

# 遍历注释文件，使用 tqdm 包装 files 来显示进度条
for root, _, files in os.walk(anno_root):
    for file in tqdm(files, desc="Processing files", unit="file"):  # 使用 tqdm 来包装文件列表
        json_path = os.path.join(root, file)
        with open(json_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)
        
        # 遍历每个 result_key 并进行统计
        for result_key in result_keys:

            if result_key.startswith("rec@iou"):
                if result_key.startswith("rec@iou_gemini-1.5_flash"):
                    miou_result_key = "miou_gemini-1.5_flash_128_True_True_True"
                elif result_key.startswith("rec@iou_gemini-1.5_pro"):
                    miou_result_key = "miou_gemini-1.5_pro_128_True_True_True"
                elif result_key.startswith("rec@iou_Qwen2-VL"):
                    miou_result_key = "miou_Qwen2-VL_72B-Instruct_128_True_True_True"
                elif result_key.startswith("rec@iou_InternVL2_5"):
                    miou_result_key = "miou_InternVL2_5_78B_32_True_True_True"
                elif result_key.startswith("rec@iou_gpt-4o_2024-08-06"):
                    miou_result_key = "miou_gpt-4o_2024-08-06_50_True_True_True"
                elif result_key.startswith("rec@iou_claude-3-5-sonnet"):
                    miou_result_key = "miou_claude-3-5-sonnet_20241022_50_True_True_True"    

                if miou_result_key in anno["results"]:
                    result_dict[result_key]["total"] += 1
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        if anno["results"][miou_result_key]["result"] >= threshold:
                            result_dict[result_key][f"iou_th_{threshold}"] += 1

            if result_key.startswith("acc@iou"):
                if result_key.startswith("acc@iou_gemini-1.5_flash"):
                    long_acc_result_key = "long_acc_gemini-1.5_flash_128_True_True_True"
                    miou_result_key = "miou_gemini-1.5_flash_128_True_True_True"
                elif result_key.startswith("acc@iou_gemini-1.5_pro"):
                    long_acc_result_key = "long_acc_gemini-1.5_pro_128_True_True_True"
                    miou_result_key = "miou_gemini-1.5_pro_128_True_True_True"
                elif result_key.startswith("acc@iou_Qwen2-VL"):
                    long_acc_result_key = "long_acc_Qwen2-VL_72B-Instruct_128_True_True_True"
                    miou_result_key = "miou_Qwen2-VL_72B-Instruct_128_True_True_True"
                elif result_key.startswith("acc@iou_InternVL2_5"):
                    long_acc_result_key = "long_acc_InternVL2_5_78B_32_True_True_True"
                    miou_result_key = "miou_InternVL2_5_78B_32_True_True_True"
                elif result_key.startswith("acc@iou_gpt-4o_2024-08-06"):
                    long_acc_result_key = "long_acc_gpt-4o_2024-08-06_50_True_True_True"
                    miou_result_key = "miou_gpt-4o_2024-08-06_50_True_True_True"
                elif result_key.startswith("acc@iou_claude-3-5-sonnet"):
                    long_acc_result_key = "long_acc_claude-3-5-sonnet_20241022_50_True_True_True"
                    miou_result_key = "miou_claude-3-5-sonnet_20241022_50_True_True_True"

                if long_acc_result_key in anno["results"] and miou_result_key in anno["results"]:
                    result_dict[result_key]["total"] += 1
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        if anno["results"][miou_result_key]["result"] >= threshold and anno["results"][long_acc_result_key]["result"] == 1:
                            result_dict[result_key][f"th_{threshold}"] += 1

            elif result_key in anno["results"]:
                result_dict[result_key]["total"] += 1

                if result_key.startswith("open"):
                    if "step_2" in anno["results"][result_key]:
                        result_dict[result_key]["result_total"] += 1
                        if anno["results"][result_key]["step_2"]["result"] == 1:
                            result_dict[result_key]["result"] += 1
                else:
                    result_dict[result_key]["result"] += anno["results"][result_key]["result"]


# 输出结果
for result_key in result_keys:

    if result_key.startswith("rec@iou"):
        print(f"Result Key: {result_key}")
        print(f"Total: {result_dict[result_key]['total']}, "
              f"iou_th_0.1: {result_dict[result_key]['iou_th_0.1']}, "
              f"iou_th_0.2: {result_dict[result_key]['iou_th_0.2']}, "
              f"iou_th_0.3: {result_dict[result_key]['iou_th_0.3']}, "
              f"iou_th_0.4: {result_dict[result_key]['iou_th_0.4']}, "
              f"iou_th_0.5: {result_dict[result_key]['iou_th_0.5']}, "
              f"Accuracy: {(result_dict[result_key]['iou_th_0.1'] + result_dict[result_key]['iou_th_0.2'] + result_dict[result_key]['iou_th_0.3'] + result_dict[result_key]['iou_th_0.4'] + result_dict[result_key]['iou_th_0.5']) / (result_dict[result_key]['total'] * 5) if result_dict[result_key]['total'] > 0 else 0}\n")        

    elif result_key.startswith("acc@iou"):
        print(f"Result Key: {result_key}")
        print(f"Total: {result_dict[result_key]['total']}, "
              f"th_0.1: {result_dict[result_key]['th_0.1']}, "
              f"th_0.2: {result_dict[result_key]['th_0.2']}, "
              f"th_0.3: {result_dict[result_key]['th_0.3']}, "
              f"th_0.4: {result_dict[result_key]['th_0.4']}, "
              f"th_0.5: {result_dict[result_key]['th_0.5']}, "
              f"Accuracy: {(result_dict[result_key]['th_0.1'] + result_dict[result_key]['th_0.2'] + result_dict[result_key]['th_0.3'] + result_dict[result_key]['th_0.4'] + result_dict[result_key]['th_0.5']) / (result_dict[result_key]['total'] * 5) if result_dict[result_key]['total'] > 0 else 0}\n")
              
    elif result_key.startswith("open"):
        # 处理 "open" 类型的结果
        print(f"Result Key: {result_key}")
        print(f"Total: {result_dict[result_key]['total']}, "
              f"Result Total: {result_dict[result_key]['result_total']}, "
              f"Result: {result_dict[result_key]['result']}, "
              f"Accuracy: {result_dict[result_key]['result'] / result_dict[result_key]['result_total'] if result_dict[result_key]['result_total'] > 0 else 0}\n")
    else:
        # 处理其他类型的结果
        print(f"Result Key: {result_key}")
        print(f"Total: {result_dict[result_key]['total']}, "
              f"Result: {result_dict[result_key]['result']}, "
              f"Accuracy: {result_dict[result_key]['result'] / result_dict[result_key]['total'] if result_dict[result_key]['total'] > 0 else 0}\n")
