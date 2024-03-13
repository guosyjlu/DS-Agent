import os
import shutil
import argparse
import numpy as np
import pandas as pd
from execution import execute_script

DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

def get_args():
    parser = argparse.ArgumentParser()
    # Model Information
    parser.add_argument("--path", default="gpt-3.5-turbo-16k_False_0")		# Code path
    parser.add_argument("--task", default="electricity")		                    # Task name
    # Generation Configuration
    parser.add_argument("--trials", default=10, type=int)    		                # Number of trials (fixed)
    # Device info
    parser.add_argument("--device", default="0", type=str)    		                # Device num
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    # Load tasks.
    if args.task == "all":
        tasks_to_evaluate = DEPLOYMENT_TASKS
    else:
        assert args.task in DEPLOYMENT_TASKS
        tasks_to_evaluate = [args.task]
    
    # Evaluate all the tasks.
    for task in tasks_to_evaluate:
        
        # Create a workspace
        work_dir = f"./workspace/{args.path}/{task}"
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            
        if os.path.exists(f"../development/MLAgentBench/benchmarks/{task}"):
            shutil.copytree(f"../development/MLAgentBench/benchmarks/{task}/env", work_dir, symlinks=True, dirs_exist_ok=True)
        
        if os.path.exists(f"./codes/{args.path}/{task}"):
            shutil.copytree(f"./codes/{args.path}/{task}", work_dir, symlinks=True, dirs_exist_ok=True)
        
        # Find submission pattern
        line = None
        with open(f"{work_dir}/submission.py") as file:
            for line in file:
                if "print" in line:
                    pattern = line
        assert line
        
        if "MSE" in line and "MAE" in line:
            results = [[], []]
        else:
            results = []
        
        pattern = line.split("\"")[1].split(":")[0]
        
        # Create result path
        result_dir = f"results/{args.path}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_filename = f"{result_dir}/{task}.csv"
        
        for idx in range(args.trials):
            filename = f"train_{idx}.py"
        
            log = execute_script(filename, work_dir=work_dir, device=args.device)
            if pattern in log:
                if "MSE" in line and "MAE" in line:
                    results[0].append(float(log.split(pattern)[1].split(":")[1].split(",")[0]))
                    results[1].append(float(log.split(pattern)[1].split(":")[2].strip(",.\n ")))
                else:
                    results.append(float(log.split(pattern)[1].split(":")[1].strip(",.\n ")))
            else:      # Fail to execute
                if "MSE" in line and "MAE" in line:
                    results[0].append(-1.0)
                    results[1].append(-1.0)
                else:
                    results.append(-1.0)
        if "MSE" in line and "MAE" in line:
            results = pd.DataFrame(results)
        else:
            results = pd.DataFrame(results).transpose()
        results.to_csv(result_filename, index=False, header=False)
        print("results")
        print("="*100)
            
        
        