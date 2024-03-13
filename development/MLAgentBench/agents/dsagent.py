""" This file contains the workflow of the proposed DS-Agent."""
import os
import sys
import anthropic
from MLAgentBench.LLM import complete_text_fast, complete_text
from MLAgentBench.schema import Action
from MLAgentBench.low_level_actions import read_file
from .agent import Agent
from .utils import clean_log


class DSAgent(Agent):

    def __init__(self, args, env):
        super().__init__(args, env)
        self.research_problem = env._research_problem
        
    def run(self, env):
        step = 0
        experiment_step = 0
        
        running_log = f"""
[Initial State] Lack of a baseline model as a good starting point for the current research problem.
"""
        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(f"Step {step}" + ":\n")
            f.write(running_log + "\n")
        
        while not env.is_final() and step < 10:
            # Develop the experiment plan (Retrieve -> RankRevise -> Reuse).
            action = "Develop An Experiment Plan via CBR"
            action_input = {
                "experiment_log": running_log
            }
            plans = env.execute(Action(action, action_input))
            step += 1
                
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + f"Action: {action}" + "\nObservation:\n" + plans + "\n") 
                
            # Execute the experiment plan (Execute)
            action = "Execute the Experiment Plan"
            action_input = {
                "script_name": "train.py",
                "plan": plans,
                "save_name": "train.py"
            }
            execution_log, diff = env.execute(Action(action, action_input))
            execution_log = clean_log(execution_log)
            step += 1
            experiment_step += 1
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + f"Action: {action}" + "\nObservation:\n" + execution_log + "\n")
            
            # Write experiment logs (Log)
            log_content = self.revise_running_log(running_log, plans, execution_log, diff, log_file=os.path.join(self.log_dir, "tmp.txt"))
            running_log += f"\n{log_content}"
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(running_log + "\n")

        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


    @staticmethod
    def revise_running_log(running_log, instructions, execution_log, diff, log_file=None):
        """ Revise progress in the running log """

        prompt = f"""Given instructions (what is expected to do), execution log (the experimental results) and the code difference (what is actually done and this will be nothing if the experiment failed) of last experiment on the research problem: 
        {instructions} 
        [Execution Log]:
        ```
        {execution_log}
        ```
        [Code Difference]:
        ```
        {diff}
        ```
        Here is the running log of your experiment:
        [Running Log]:
        ```
        {running_log}
        ```
        Summarize and append the progress of the last step to the running log in this format:
        [Experiment Summary]: According to the instructions and the code difference, summarize what was experimented in the last step objectively.
        [Experiment Result]: According to the execution log and the running log, summarize if the last step of experiment brings performance improvement objectively. Only report the performance if this is the first experiment result.
        Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        log = "[Experiment Summary]:" + complete_text_fast(prompt, log_file=log_file).split("[Experiment Summary]:")[1]
        return log

    