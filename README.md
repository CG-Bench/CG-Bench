# CG-Bench

This repository contains the implementation of CG-Bench. Follow the steps below to set up and run the benchmark.  
Project Website: https://cg-bench.github.io/leaderboard/  
Huggingface Link: https://huggingface.co/datasets/CG-Bench/CG-Bench

## Setup and Data Preparation

1. Clone the repository:
```bash
git clone https://github.com/CG-Bench/CG-Bench.git
cd CG-Bench
```

2. Download and unzip the dataset:
```bash
python unzip_hf_zip.py
```

3. Process the JSON files:
```bash
python run/save_as_jsons.py
```

## Testing

4. Before running the test, make sure to configure your API credentials in `run/run_api.py`:
   - Set your `api_base`
   - Set your `api_key`

5. Run the test script:
```bash
bash run.sh clue_acc gpt-4o 2024-08-06 32 true true true # (or long_acc, miou, open ...) 
```

## View Results

6. Check the test results:
```bash
python stat_with_key.py
```

## Note
Make sure you have properly configured your API credentials in `run/run_api.py` before running the tests. Without valid API credentials, the tests will fail.
