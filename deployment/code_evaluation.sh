#### GPT-3.5

# Zero-shot
nohup python -u evaluation.py --path "gpt-3.5-turbo-16k_False_0" --task "all" --device 0 > eval_zero_shot.txt 2>&1 &

# DS-Agent
nohup python -u evaluation.py --path "gpt-3.5-turbo-16k_True_1" --task "all" --device 1  > eval_cbr.txt 2>&1 &

# Randomly-Retrieved One-shot
nohup python -u evaluation.py --path "gpt-3.5-turbo-16k_False_1" --task "all" --device 2  > eval_random_cbr.txt 2>&1 &

# Raw Case
nohup python -u evaluation.py --path "gpt-3.5-turbo-16k_True_1_raw" --task "all" --device 3  > eval_raw3.txt 2>&1 &

#### GPT-4

# Zero-shot
nohup python -u evaluation.py --path "gpt-4_False_0" --task "all" --device 0 > eval_zero_shot_4.txt 2>&1 &

# DS-Agent
nohup python -u evaluation.py --path "gpt-4_True_1" --task "all" --device 1  > eval_cbr_4.txt 2>&1 &

# Randomly-Retrieved One-shot
nohup python -u evaluation.py --path "gpt-4_False_1" --task "all" --device 2  > eval_random_cbr_4.txt 2>&1 &

# Raw Case
nohup python -u evaluation.py --path "gpt-4_True_1_raw" --task "all" --device 3  > eval_raw4.txt 2>&1 &