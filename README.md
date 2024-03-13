# DS-Agent

This is the official implementation of our work "DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning". [[arXiv Version]](https://arxiv.org/abs/2402.17453) [[Download Benchmark(Google Drive)]](https://drive.google.com/file/d/1xUd1nvCsMLfe-mv9NBBHOAtuYnSMgBGx/view?usp=sharing)

![overview.png](.\figures\overview.png)

## Benchmark

We select 30 representative data science tasks covering three data modalities and two fundamental ML task types. Please download the datasets and corresponding configuration files via [[Google Drive]](https://drive.google.com/file/d/1xUd1nvCsMLfe-mv9NBBHOAtuYnSMgBGx/view?usp=sharing)  here and unzip them to the directory of "development/benchamarks".

![overview.png](.\figures\task.png)

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

## Cite

Please consider citing our paper if you find this work useful:

```
@article{DS-Agent,
  title={DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning},
  author={Guo, Siyuan and Deng, Cheng and Wen, Ying and Chen, Hechang and Chang, Yi and Wang, Jun},
  journal={arXiv preprint arXiv:2402.17453},
  year={2024}
}
```
