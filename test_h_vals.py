import torch
import numpy as np
from matplotlib import pyplot as plt

def main():
    try:
        data = torch.load("gcbf_swarm_checkpoint.pt", map_location="cpu", weights_only=False)
        gcbf = data["gcbf_net"]
        print("Loaded GCBF")
        
        # Create a synthetic state with 3 agents
        states = torch.zeros(1, 3, 4)
        states[0, 0, :2] = torch.tensor([2.0, 2.0]) # agent 0
        states[0, 1, :2] = torch.tensor([3.0, 3.0]) # agent 1
        states[0, 2, :2] = torch.tensor([1.0, 3.0]) # agent 2
        
        obs = torch.zeros(1, 1, 4)
        obs[0, 0, :2] = torch.tensor([2.0, 1.0]) # obstacle at (2,1)
        obs[0, 0, 2:4] = torch.tensor([0.2, 0.2]) # obstacle half size
        
        from gcbf_plus.utils.swarm_graph import build_vectorized_swarm_graph
        graph = build_vectorized_swarm_graph(states, states, obs, 2.0, 3, 4)
        
        h = gcbf(graph).squeeze(-1)
        print("h vals:", h)
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
