import os
import zipfile
from tqdm import tqdm
from huggingface_hub import snapshot_download

def modelscope_flag_set():
    """Check if modelscope should be used"""
    try:
        from modelscope import dataset_snapshot_download
        return True
    except ImportError:
        return False

def get_chunk_number(filename):
    try:
        num = filename.split("chunk_")[1].split(".zip")[0]
        return int(num)
    except:
        return float('inf')

def unzip_hf_zip(pth):

    download_dir = pth
    target_dir = "."

    if os.path.exists(f"{target_dir}/cg_videos_720p") and os.path.exists(f"{target_dir}/cg_subtitles")\
            and os.path.exists(f"{target_dir}/cg_clue_videos"):
        print("all exists")
        return

    video_zip_files = [
        os.path.join(download_dir, file) for file in os.listdir(download_dir)
        if file.endswith('.zip') and file.startswith('video')
    ]

    video_zip_files = sorted(video_zip_files, key=lambda x: get_chunk_number(os.path.basename(x)))

    videos_temp_zip = os.path.join(download_dir, "videos_merged.zip")

    print("Merging video files ...")

    with open(videos_temp_zip, 'wb') as outfile:
        for video_zip_file in tqdm(video_zip_files, desc="Merging videos"):
            with open(video_zip_file, 'rb') as infile:
                outfile.write(infile.read())

    print("Extracting video files...")

    try:
        with zipfile.ZipFile(videos_temp_zip, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())

            for file in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                zip_ref.extract(file, target_dir)

        print(f"Successfully extracted to {target_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:
        if os.path.exists(videos_temp_zip):
            os.remove(videos_temp_zip)
            print("Cleaned up temporary video file")

    clue_video_zip_files = [
        os.path.join(download_dir, file) for file in os.listdir(download_dir)
        if file.endswith('.zip') and file.startswith('clue_video')
    ]

    clue_video_zip_files = sorted(clue_video_zip_files, key=lambda x: get_chunk_number(os.path.basename(x)))

    clue_videos_temp_zip = os.path.join(download_dir, "clue_videos_merged.zip")

    print("Merging clue video files ...")

    with open(clue_videos_temp_zip, 'wb') as outfile:
        for clue_video_zip_file in tqdm(clue_video_zip_files, desc="Merging clue_videos"):
            with open(clue_video_zip_file, 'rb') as infile:
                outfile.write(infile.read())

    print("Extracting clue video files...")

    try:
        with zipfile.ZipFile(clue_videos_temp_zip, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())

            for file in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                zip_ref.extract(file, target_dir)

        print(f"Successfully extracted to {target_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:
        if os.path.exists(clue_videos_temp_zip):
            os.remove(clue_videos_temp_zip)
            print("Cleaned up temporary clue video file")

    print("Extracting subtitle files ...")

    subtitles_zip = os.path.join(download_dir, "subtitles.zip")

    try:
        with zipfile.ZipFile(subtitles_zip, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())

            for file in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                zip_ref.extract(file, target_dir)

        print(f"Successfully extracted to {target_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    repo_id = "CG-Bench/CG-Bench"

    if modelscope_flag_set():
        from modelscope import dataset_snapshot_download
        dataset_path = dataset_snapshot_download(dataset_id=repo_id)
    else:
        dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

    unzip_hf_zip(dataset_path)