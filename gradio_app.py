import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import gradio as gr
import os
import numpy as np
from torchvision import transforms

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset

# Load the model
class Model:
    def __init__(self):
        self.model = BCResNets(int(self.tau * 8)).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))       

def classify():
    #TODO: Implement the forward pass
    pass 

interface = gr.Interface(
    fn=classify,
    inputs=gr.Audio(),
    outputs="label",
    title="Speech Command Recognition",
    description="Recognize speech commands using a ResNet model.",
    theme="huggingface",
)
interface.launch()