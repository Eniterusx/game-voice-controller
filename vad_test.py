import torch
from torchvision import transforms

import queue
import sys
import argparse

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import webrtcvad
import collections
from pathlib import Path

from bcresnet import BCResNets
from utils import Padding, Preprocess

import matplotlib
matplotlib.use('TkAgg')

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
        
class VADGUI:
    def __init__(self, device):
        self.sample_rate = 16000
        self.frame_duration = 30
        self.chunk_duration = 1000
        self.overlap_duration = 700
        self.vad_aggressiveness = 2
        self.tau = 3
        self.model_path = "model.pth"

        self.callback = self.update_detection_plot

        self.model = Model(self.tau, self.model_path)
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

        self.frame_length = int(self.sample_rate * self.frame_duration / 1000)
        self.chunk_length = int(self.sample_rate * self.chunk_duration / 1000)
        self.overlap_length = int(self.sample_rate * self.overlap_duration / 1000)
        self.circular_buffer = collections.deque(maxlen=self.chunk_length + self.overlap_length)

        self.channels = [1]
        self.mapping = [c - 1 for c in self.channels]  # Channel numbers start with 1
        self.device = device
        self.window = 400
        self.interval = 30
        self.blocksize = 800
        self.downsample = 10

        self.last_input = None

        self.setup()
        self.update_detection_plot("")
        self.loop()

    def setup(self):
        try:
            self.q = queue.Queue()

            self.length = int(self.window * self.sample_rate / (1000 * self.downsample))
            self.plotdata = np.zeros((self.length, len(self.channels)))

            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
            self.fig = fig
            self.lines = ax1.plot(self.plotdata)
            ax1.axis((0, len(self.plotdata), -1, 1))
            ax1.set_yticks([0])
            ax1.yaxis.grid(True)
            ax1.tick_params(bottom=False, top=False, labelbottom=False,
                            right=False, left=False, labelleft=False)
            self.fig.tight_layout(pad=0)

            self.status_ax = ax2
            self.status_ax.set_xlim(0, 1)
            self.status_ax.set_ylim(0, 1)
            self.status_ax.axis('off')
            self.text_artist = self.status_ax.text(
                0.5, 0.5, '', 
                horizontalalignment='center', 
                verticalalignment='center', 
                fontsize=32, 
                color='red'
            )
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))

    def loop(self):
            try:
                stream = sd.InputStream(
                    channels=1, samplerate=self.sample_rate, device=self.device,
                    dtype='int16', blocksize=self.blocksize, callback=self.audio_callback)

                plot_update_interval = 30
                ani = FuncAnimation(self.fig, self.update_plot, interval=plot_update_interval, blit=True)
                ani_detection = FuncAnimation(self.fig, self.update_detection_plot, interval=plot_update_interval, blit=True)
                with stream:
                    plt.show()
            except Exception as e:
                print(type(e).__name__ + ': ' + str(e))

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
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        q_data = indata.astype(np.float32) / 32768.0
        self.q.put(q_data[::self.downsample, self.mapping])
        # circular_buffer.append(indata[::args.downsample, mapping])

        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        self.circular_buffer.extend(audio_chunk)

        while len(self.circular_buffer) >= self.chunk_length + self.overlap_length:
            full_chunk = list(self.circular_buffer)[self.overlap_length:]
            frames = self._process_audio_chunk(full_chunk)

            # check last_input for debouncing
            if self._is_speech_chunk(frames) and self.last_input is None:
                command = self.model.classify(full_chunk)
                if command != "_silence_" and command != "_unknown_":
                    self.callback(command)
                self.last_input = command
                print(self.last_input.replace("_", "").capitalize(), end="", flush=True)
            else:
                self.callback(None)
                self.last_input = None
                print(".", end="", flush=True)

            for _ in range(self.overlap_length):
                self.circular_buffer.popleft()

    def update_plot(self, frame):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """
        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines

    def update_detection_plot(self, frame):
        if self.last_input is None:
            text = "No Input"
            color = "red"
        elif self.last_input in ["_silence_", "_unknown_"]:
            text = self.last_input.replace("_", "").capitalize()
            color = "red"
        else:
            text = self.last_input.capitalize()
            color = "green"
        # check if attribute exists
        if not hasattr(self, 'text_artist'):
            return

        self.text_artist.set_text(text)
        self.text_artist.set_color(color)
        
        return [self.text_artist]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Inference with VAD")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    else:
        vadgui = VADGUI(device=args.device)
