import os
import re
import torch
import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from MLAgentBench.LLM import complete_text

RANKING_MODEL = "gpt-3.5-turbo-16k"

class RetrievalDatabase:
    def __init__(self, dirList, model="BAAI/llm-embedder", batch_size=32) -> None:
        self.dirList = dirList
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True).to(self.device)
        
        # Define query
        if model == "BAAI/llm-embedder":
            self.query_prompt = "Represent this query for retrieving relevant documents: "
            self.doc_prompt = "Represent this document for retrieval: "
        else:
            self.query_prompt = ""
            self.doc_prompt = ""
        
        # Read cases
        self.case_bank = []
        for dirname in self.dirList:
            fileList = os.listdir(dirname)
            for filename in fileList:
                with open(f"{dirname}/{filename}") as file:
                    self.case_bank.append(self.query_prompt + file.read())
        
        # Construct Embedding Database
        if model == "BAAI/llm-embedder":
            x_inputs = self.tokenizer(
                self.case_bank,
                padding=True, 
                truncation= True,
                return_tensors='pt'
            )
            
            input_ids = x_inputs.input_ids.to(self.device)
            attention_mask = x_inputs.attention_mask.to(self.device)

            with torch.no_grad():
                x_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                x_outputs = x_outputs.last_hidden_state[:, 0]
                x_embedding = torch.nn.functional.normalize(x_outputs, p=2, dim=1)
            
            self.embedding_bank = x_embedding
        else:
            raise NotImplementedError("The retriever is not implemented yet.")
            
    
    def retrieve_case(self, query, num=10):
        x_inputs = self.tokenizer(
            self.query_prompt + query,
            padding=True, 
            truncation= True,
            return_tensors='pt'
        )
        input_ids = x_inputs.input_ids.to(self.device)
        attention_mask = x_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            x_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            x_outputs = x_outputs.last_hidden_state[:, 0]
            x_embedding = torch.nn.functional.normalize(x_outputs, p=2, dim=1)
        
        similarity = (x_embedding @ self.embedding_bank.T).squeeze()         
        _, ranking_index = torch.topk(similarity, num)

        ranking_index = ranking_index.cpu().numpy().tolist()
        if self.query_prompt:
            return [self.case_bank[i].split(self.query_prompt)[1] for i in ranking_index], similarity[ranking_index]
        else:
            return [self.case_bank[i] for i in ranking_index], similarity[ranking_index]
    
    def retrieve_then_rerank(self, query, research_problem, research_log, log_file, topk=5):
        # Retriever
        case_bank, _ = self.retrieve_case(query, num=topk)
        # RankReviser
        prompt = f"""
You are a helpful intelligent system that can identify the informativeness of some cases given a research problem and research log.
Research Problem: ```
{research_problem}
```
Research Log: ```
{research_log}
```
Here are some solution cases relevant to this research problem, each indicated by number identifier [].
[1] ```
{case_bank[0]}
```
[2] ```
{case_bank[1]}
```
[3] ```
{case_bank[2]}
```
[4] ```
{case_bank[3]}
```
[5]```
{case_bank[4]}
```
Rank 5 cases above based on their relevance, informativess and helpfulness to the research problem and the research log for planning the next experiment step. The cases should be listed in descending order using identifiers. The most relevant, informative and helpful case should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.
"""
        ranking = complete_text(prompt, model=RANKING_MODEL, log_file=log_file)
        ranking = re.findall(r'\[(\d+)\]', ranking)
        ranking = [int(num)-1 for num in ranking]
        return case_bank[ranking[0]]