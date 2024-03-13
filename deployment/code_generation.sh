#### GPT-3.5
# Zero-shot
nohup python -u generate.py --llm gpt-3.5-turbo-16k --task all --shot 0 > zero.txt 2>&1 &

# DS-Agent
nohup python -u generate.py --llm gpt-3.5-turbo-16k --task all --shot 1 --retrieval > cbr.txt 2>&1 &

# One-shot
nohup python -u generate.py --llm gpt-3.5-turbo-16k --task all --shot 1 > random-cbr.txt 2>&1 &

# Raw case
nohup python -u generate.py --llm gpt-3.5-turbo-16k --task all --shot 1 --retrieval --raw > raw.txt 2>&1 &

#### GPT-4
# Zero-shot
nohup python -u generate.py --llm gpt-4 --task all --shot 0 > zero4.txt 2>&1 &

# DS-Agent
nohup python -u generate.py --llm gpt-4 --task all --shot 1 --retrieval > cbr4.txt 2>&1 &

# One-shot
nohup python -u generate.py --llm gpt-4 --task all --shot 1 > random-cbr4.txt 2>&1 &

# Raw case
nohup python -u generate.py --llm gpt-4 --task all --shot 1 --retrieval --raw > raw4.txt 2>&1 &