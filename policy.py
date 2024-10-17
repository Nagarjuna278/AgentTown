import torch
from torch.distributions import Categorical
import random
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        #self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,output_size)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

    @staticmethod
    def load_model(path, input_size, output_size):
        model = PolicyNetwork(input_size, output_size)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model    