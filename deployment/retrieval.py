import os
import re
import json
import torch
import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

DEVELOPMENT_TASKS = ["feedback", "airline-reviews", "textual-entailment", "chatgpt-prompt", "ett-m2", "ili", "handwriting", "ethanol-concentration", "media-campaign-cost", "wild-blueberry-yield", "spaceship-titanic", "enzyme-substrate"]
DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

class RetrievalDatabase:
    def __init__(self, model="BAAI/llm-embedder") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        for task in DEVELOPMENT_TASKS:
            filename = f"../development/MLAgentBench/benchmarks/{task}/scripts/research_problem.txt"
            with open(filename) as file:
                self.case_bank.append(self.query_prompt + file.read())
        
        # Construct Embedding Database
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
            
    
    def retrieve_case(self, query, num=12):
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
        
        return [DEVELOPMENT_TASKS[i] for i in ranking_index]
    

if __name__ == '__main__':
    rb = RetrievalDatabase()
    ranking_dict = {}
    for task in DEPLOYMENT_TASKS:
        filename = f"../development/MLAgentBench/benchmarks/{task}/scripts/research_problem.txt"
        with open(filename) as file:
            query = file.read()
            doc = rb.retrieve_case(query)
        ranking_dict[task] = doc
    with open("config/similaity_ranking.json", "wt") as json_file:
        json.dump(ranking_dict, json_file, indent=4)
        
