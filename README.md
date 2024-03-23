# DS-Agent

This is the official implementation of our work "DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning".

![overview.png](figures/overview.png)

## Benchmark and Dataset

We select 30 representative data science tasks covering three data modalities and two fundamental ML task types. We will open-source these datasets after publication.

![overview.png](figures/task.png)

## Setup

This project is built on top of the framework of MLAgentBench. First, install MLAgentBench package with:

```shell
cd development
pip install -e.
```

Then, please install neccessary libraries in the requirements.

```shell
pip install -r requirements.txt
```

Since DS-Agent mainly utilizes GPT-3.5 and GPT-4 for all the experiments, please fill in the openai key in development/MLAgentBench/LLM.py and deployment/generate.py

## Development Stage

Run DS-Agent for development tasks with the following command:

```shell
cd development/MLAgentBench
python runner.py --task feedbackv2 --llm-name gpt-3.5-turbo-16k --edit-script-llm-name gpt-3.5-turbo-16k
```

During execution, logs and intermediate solution files will be saved in logs/ and workspace/. 

## Deployment Stage

Run DS-Agent for deployment tasks with the provided command:

```shell
cd deployment
bash code_generation.sh
bash code_evaluation.sh
```

For open-sourced LLM, i.e., mixtral-8x7b-Instruct-v0.1 in this paper, we utilize the vllm framework. First, enable the LLMs serverd with

```shell
cd deployment
bash start_api.sh
```

Then, run the script shell and replace the configuration --llm by mixtral.
