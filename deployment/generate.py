import openai
import argparse
import torch
import os
import anthropic
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import get_prompt

openai.api_key = "FILL IN YOUR KEY HERE."
# openai.api_base = "http://localhost:8000/v1"
enc = tiktoken.get_encoding("cl100k_base")

DEVELOPMENT_TASKS = ["feedback", "airline-reviews", "textual-entailment", "chatgpt-prompt", "ett-m2", "ili", "handwriting", "ethanol-concentration", "media-campaign-cost", "wild-blueberry-yield", "spaceship-titanic", "enzyme-substrate"]
DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

def get_args():
    parser = argparse.ArgumentParser()
    # Model Information
    parser.add_argument("--llm", default="gpt-3.5-turbo-16k")		# LLM name
    parser.add_argument("--task", default="detect-ai-generation")   # ML Task name
    # Context Configuration
    parser.add_argument("--shot", default=0, type=int)              # Number of examples in context
    parser.add_argument("--retrieval", default=False,               # Whether activate retrieval
                        action='store_true')
    parser.add_argument("--raw", default=False,                     # Whether use raw cases
                        action='store_true')
    # Generation Configuration
    parser.add_argument("--temperature", default=0.7, type=float)   # Temperature (fixed)
    parser.add_argument("--trials", default=10, type=int)    		# Number of trials (fixed)

    args = parser.parse_args()
    return args

def generation(prompt, llm, temperature=0.7, log_file=None):
    raw_request = {
        "model": llm,
        "temperature": temperature,
        "max_tokens": 1500,
        "stop": [] or None,
    }
    iteration = 0
    completion = None
    while iteration < 50:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(**{"messages": messages,**raw_request})
            raw_completion = response["choices"][0]["message"]["content"]
            completion = raw_completion.split("```python")[1].split("```")[0]
            if not completion.strip(" \n"):
                continue
            break
        except Exception as e:
            iteration += 1
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
        continue
    if not completion:
        completion = ""
    print(completion)
    log_to_file(log_file, prompt, raw_completion)
    return completion
    

def log_to_file(log_file, prompt, completion):
    """ Log the prompt and completion to a file."""
    num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
    num_sample_tokens = len(enc.encode(completion))
    
    # Logging for finetuning
    with open(log_file, "wt") as f:
        f.write(prompt)
        f.write("\n[This is a split string for finetuning]\n")
        f.write(completion)
        f.write("\n[This is a split string for counting tokens]\n")
        f.write(f"Prompt: {num_prompt_tokens}, Completion: {num_sample_tokens}")


if __name__ == '__main__':
    args = get_args()
    
    # Load Model
    if "mixtral" in args.llm:
        openai.api_base = "http://localhost:8000/v1"
        
    # Load Tasks
    if args.task == "all":
        tasks_to_solve = DEPLOYMENT_TASKS
    else:
        assert args.task in DEPLOYMENT_TASKS
        tasks_to_solve = [args.task]
    
    # Pathname
    prefix = f"{args.llm}_{args.retrieval}_{args.shot}" if not args.raw else f"{args.llm}_{args.retrieval}_{args.shot}_raw"
    
    # Create the path for generation results
    pathname = f"./codes/{prefix}"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
       
    # Create Finetune Logs
    finetune_dir = f"./codes/{prefix}/finetune_log"
    if not os.path.exists(finetune_dir):
        os.makedirs(finetune_dir)
    
    for task in tasks_to_solve:
        print(f"Processing Task: {task}")
        tmp_pathname = f"{pathname}/{task}"
        if not os.path.exists(tmp_pathname):
            os.makedirs(tmp_pathname)
        temp_finetunedir = f"{finetune_dir}/{task}"
        if not os.path.exists(temp_finetunedir):
            os.makedirs(temp_finetunedir)
        for idx in range(args.trials):
            prompt = get_prompt(task, context_num=args.shot, strategy="retrieval" if args.retrieval else "random", raw=args.raw)
            response = generation(prompt, args.llm, temperature=args.temperature, log_file=f"{temp_finetunedir}/{idx}.txt")
            filename = f"{tmp_pathname}/train_{idx}.py"
            with open(filename, "wt") as file:
                file.write(response)
            
        