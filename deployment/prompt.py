import json
import random

RP_PATH = "./benchmarks/{}/scripts/research_problem.txt"
PYTHON_PATH = "./benchmarks/{}/env/train.py"
CASE_PATH = "./experience_replay/{}.py"

ZERO_SHOT_PROMPT = """
You are a helpful intelligent assistant. Now please help solve the following machine learning task.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
"""

FEW_SHOT_PROMPT = """
Here are some example cases that solve machine learning tasks:
{} 
Now please solve the following machine learning task based on the example cases above.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
"""

CASE_PROMPT = """[Task]
{}
[train.py] ```python
{}
```
[Solution] ```python
{}
```
"""

RAW_CASE_PROMPT = """Here are some relevant textual insights that can hel you solve the machine learning task:
{} 
Now please solve the following machine learning task based on the textual insights above.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
```
"""

def get_task(task):
    rp_path = RP_PATH.format(task)
    python_path = PYTHON_PATH.format(task)
    with open(rp_path) as file:
        rp = file.read()
    with open(python_path) as file:
        code = file.read()
    return rp, code

def get_case(task):
    rp_path = RP_PATH.format(task)
    python_path = PYTHON_PATH.format(task)
    case_path = CASE_PATH.format(task)
    with open(rp_path) as file:
        rp = file.read()
    with open(python_path) as file:
        code = file.read()
    with open(case_path)as file:
        case = file.read()
    return CASE_PROMPT.format(rp, code, case)

def get_prompt(task, context_num=0, strategy=None, raw=False):
    rp, code = get_task(task)
    
    # Ablation Study
    if raw:
        with open("./config/heterogenous_similarity_ranking.json") as file:
            ranking_dictionary = json.load(file)
            case = ranking_dictionary[task]
            return RAW_CASE_PROMPT.format(case, rp, code)
    if context_num == 0:
        return ZERO_SHOT_PROMPT.format(rp, code)
    else:
        with open("./config/similarity_ranking.json") as file:
            ranking_dictionary = json.load(file)
        if strategy == "retrieval":
            selected_tasks = ranking_dictionary[task][:context_num]
        elif strategy == "random":
            selected_tasks = random.sample(ranking_dictionary[task], k=context_num)
        else:
            raise NotImplementedError("This strategy is not supported yet!")
        examples = ""
        for i in selected_tasks:
            examples += get_case(i)
        return FEW_SHOT_PROMPT.format(examples, rp, code)
        

if __name__ == '__main__':
    p = get_prompt("cirrhosis-outcomes", context_num=1, strategy="retrieval", raw_case=True)
    print(p)