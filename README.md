# DS-Agent

This is the official implementation of our work "DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning" (ICML 2024). [[arXiv Version]](https://arxiv.org/abs/2402.17453) [[Download Benchmark(Google Drive)]](https://drive.google.com/file/d/1xUd1nvCsMLfe-mv9NBBHOAtuYnSMgBGx/view?usp=sharing)

![overview.png](figures/overview.png)

## Benchmark and Dataset

We select 30 representative data science tasks covering three data modalities and two fundamental ML task types. Please download the datasets and corresponding configuration files via [[Google Drive]](https://drive.google.com/file/d/1xUd1nvCsMLfe-mv9NBBHOAtuYnSMgBGx/view?usp=sharing)  here and unzip them to the directory of "development/benchmarks". Besides, we collect the human insight cases from Kaggle in development/data.zip. Please unzip it, too.

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
python runner.py --task feedback --llm-name gpt-3.5-turbo-16k --edit-script-llm-name gpt-3.5-turbo-16k
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

## Frequently Asked Questions

### Q1. How to calculate the best rank and mean rank of the evaluated agents?
**A1.** Assume there are two agents A and B. Given a data science task, both agents perform 5 random trials to build models. Then, we use the predefined evaluation metric to evaluate the built model in the testing set. As such, we can rank these ten built models via the evaluation results.
> Assume the models built by Agent A attains the rank [1,3,5,7,9], and the models built by Agent B attains the rank [2,4,6,8,10].

As such, MeanRank(A)=mean([1,3,5,7,9])=5, BestRank(A)=min([1,3,5,7,9])=1. Similarly, MeanRank(B)=6, BestRank(B)=2.

## Cite

Please consider citing our paper if you find this work useful:

```

@InProceedings{DS-Agent,
  title = 	 {{DS}-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning},
  author =       {Guo, Siyuan and Deng, Cheng and Wen, Ying and Chen, Hechang and Chang, Yi and Wang, Jun},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {16813--16848},
  year = 	 {2024},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
}

```
