import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import gradio as gr
import numpy as np
from torchvision import transforms
from pathlib import Path

from bcresnet import BCResNets
from utils import Padding, Preprocess

label_dict = {
    0: "_silence_",
    1: "_unknown_",
    2: "down",
    3: "go",
    4: "left",
    5: "no",
    6: "off",
    7: "on",
    8: "right",
    9: "stop",
    10: "up",
    11: "yes",
    12: "zero",
    13: "one",
    14: "two",
    15: "three",
    16: "four",
    17: "five",
    18: "six",
    19: "seven",
    20: "eight",
    21: "nine",
}

# Load the model
class Model:
    #TODO: Update for higher self.tau (maybe 2)
    def __init__(self):
        self.tau = 1
        self.gpu = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BCResNets(int(self.tau * 8)).to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()
        self.noise_dir = Path("./data/speech_commands_v0.02_split/_background_noise_")
        self.preprocess = Preprocess(noise_loc=None, device=self.device)
        self.transform = transforms.Compose([Padding()]) # zero pad to have 1 sec len

    def classify(self, audio):
        sample = audio[1] / 32768.0 # convert to range -1 to 1
        sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        with torch.no_grad():
            sample = sample.unsqueeze(0).unsqueeze(0) # unsqueeze to add two dimensions
            sample = sample.to(self.device)
            sample = self.preprocess(sample, None, augment=False)
            output = self.model(sample)
            pred = output.argmax(dim=1, keepdim=True)
            return label_dict[pred.item()]

model = Model()

interface = gr.Interface(
    fn=model.classify,
    inputs=gr.Audio(),
    outputs="label",
    title="Speech Command Recognition",
    description="Recognize speech commands using a BC-ResNet model.",
    theme="huggingface",
)
interface.launch()