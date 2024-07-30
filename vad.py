import torch
from torchvision import transforms

import numpy as np
import sounddevice as sd
import webrtcvad
import collections
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

class Model:
    def __init__(self, tau=3, model_path="model.pth"):
        self.tau = tau
        self.gpu = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BCResNets(int(self.tau * 8)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.noise_dir = Path("./data/speech_commands_v0.02_split/_background_noise_")
        self.preprocess = Preprocess(noise_loc=None, device=self.device)
        self.transform = transforms.Compose([Padding()]) # zero pad to have 1 sec len

    def classify(self, audio):
        sample = np.array(audio) / 32768.0 # convert to range -1 to 1
        sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        with torch.no_grad():
            sample = sample.unsqueeze(0).unsqueeze(0) # unsqueeze to add two dimensions
            sample = sample.to(self.device)
            sample = self.preprocess(sample, None, augment=False)
            output = self.model(sample)
            pred = output.argmax(dim=1, keepdim=True)
            return label_dict[pred.item()]


class VADModel:
    def __init__(self, callback):
        self.sample_rate = 16000
        self.frame_duration = 30
        self.chunk_duration = 1000
        self.overlap_duration = 700
        self.vad_aggressiveness = 2
        self.tau = 3
        self.model_path = "model.pth"
        self.device_id = None

        self.callback = callback

        self.model = Model(self.tau, self.model_path)
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        self.chunk_length = int(self.sample_rate * self.chunk_duration / 1000)
        self.overlap_length = int(self.sample_rate * self.overlap_duration / 1000)
        self.circular_buffer = collections.deque(maxlen=self.chunk_length + self.overlap_length)

    def _process_audio_chunk(self, audio_chunk):
        frames = []
        for start in range(0, len(audio_chunk), self.frame_length):
            frames.append(audio_chunk[start:start + self.frame_length])
        return frames
    
    def _is_speech_chunk(self, frames):
        speech_count = 0
        for frame in frames:
            if len(frame) == self.frame_length:
                frame = np.array(frame, dtype=np.int16)
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    speech_count += 1
        speech_ratio = speech_count / len(frames)
        return speech_ratio >= 0.2
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        
        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        self.circular_buffer.extend(audio_chunk)
        
        while len(self.circular_buffer) >= self.chunk_length + self.overlap_length:
            full_chunk = list(self.circular_buffer)[self.overlap_length:]
            frames = self._process_audio_chunk(full_chunk)
            
            if self._is_speech_chunk(frames):
                command = self.model.classify(full_chunk)
                if command != "_silence_" and command != "_unknown_":
                    self.callback(command)

            for _ in range(self.overlap_length):
                self.circular_buffer.popleft()