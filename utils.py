# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import random
from glob import glob
import shutil
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

### GSC
label_dict = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
    "zero": 12,
    "one": 13,
    "two": 14,
    "three": 15,
    "four": 16,
    "five": 17,
    "six": 18,
    "seven": 19,
    "eight": 20,
    "nine": 21,
}
print("labels:\t", label_dict)
sample_per_cls_v1 = [1854, 258, 257]
sample_per_cls_v2 = [3077, 371, 408]
SR = 16000

def SplitCommands(seed=42):
    '''
    Used in finetuning to split the commands into two dictionaries,
    one for pretraining and one for finetuning.
    '''
    random.seed(seed)

    pretrain_commands = 15
    commands = [x for x in label_dict.keys() if x not in ("_silence_", "_unknown_")]
    random.shuffle(commands)

    pretrain_dict = {command: i+2 for i, command in enumerate(commands[:pretrain_commands])}
    pretrain_dict["_silence_"] = 0
    pretrain_dict["_unknown_"] = 1
    pretrain_dict = dict(sorted(pretrain_dict.items(), key=lambda item: item[1]))

    finetune_dict = {command: i+2 for i, command in enumerate(commands[pretrain_commands:])}
    finetune_dict["_silence_"] = 0
    finetune_dict["_unknown_"] = 1
    finetune_dict = dict(sorted(finetune_dict.items(), key=lambda item: item[1]))

    print("pretrain_dict:\t", pretrain_dict)
    print("finetune_dict:\t", finetune_dict)

    return pretrain_dict, finetune_dict


def GetAudioPaths(root_dir, command, seed=42, percentage=1.0):
    '''
    Get the audio paths for a specific command
    '''
    random.seed(seed)

    audio_paths = []

    files_per_command = int(1500 * percentage) # 1500 files per command by default

    files = [file for file in os.listdir(os.path.join(root_dir, command)) if file.endswith(".wav")]
    random.shuffle(files)
    for file in files[:files_per_command]:
        audio_paths.append(os.path.join(root_dir, command, file))
    
    return audio_paths


def SplitDataset(root_dir, seed=42, percentage=1.0):
    '''
    Split the dataset into training, validation and test
    '''
    random.seed(seed)

    train_files = int(1500 * percentage * 0.7)
    valid_files = int(1500 * percentage * 0.15)
    test_files = int(1500 * percentage * 0.15)

    # round up or down to make sure the total amount of files is equal to the desired amount
    while train_files + valid_files + test_files > int(1500 * percentage):
        train_files -= 1
    while train_files + valid_files + test_files < int(1500 * percentage):
        train_files += 1

    labels_dict = {command: i for i, command in enumerate(label_dict.keys())}
    train_data, valid_data, test_data = [], [], []

    for command in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, command)):
            continue
        if command == "_background_noise_":
            continue
        audio_paths = GetAudioPaths(root_dir, command, seed, percentage)
        if command not in labels_dict.keys():
            command = "_unknown_"
        labels = [labels_dict[command]] * len(audio_paths)
        train_data += zip(audio_paths[:train_files], labels[:train_files])
        valid_data += zip(audio_paths[train_files:train_files + valid_files], labels[train_files:train_files + valid_files])
        test_data += zip(audio_paths[train_files + valid_files:], labels[train_files + valid_files:])
    
    # add silence using the same amount of files as the other commands
    # use make_empty_audio to create the silence file
    make_empty_audio(root_dir)
    for _ in range(train_files):
        train_data.append((os.path.join(root_dir, "_silence_.wav"), 0))
    for _ in range(valid_files):
        valid_data.append((os.path.join(root_dir, "_silence_.wav"), 0))
    for _ in range(test_files):
        test_data.append((os.path.join(root_dir, "_silence_.wav"), 0))
    data = {"train": train_data, "valid": valid_data, "test": test_data}
    return data, labels_dict


def FineTuneSplit(root_dir, seed=42, percentage=1.0):
    '''
    Split the dataset into pretraining and finetuning data
    '''
    random.seed(seed)

    train_files = int(1500 * percentage * 0.7)
    valid_files = int(1500 * percentage * 0.15)
    test_files = int(1500 * percentage * 0.15)

    # round up or down to make sure the total amount of files is equal to the desired amount
    while train_files + valid_files + test_files > int(1500 * percentage):
        train_files -= 1
    while train_files + valid_files + test_files < int(1500 * percentage):
        train_files += 1

    pretrain_dict, finetune_dict = SplitCommands(seed)
    pretrain_train, pretrain_eval, pretrain_test = [], [], []
    finetune_train, finetune_eval, finetune_test = [], [], []

    for command in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, command)):
            continue
        if command == "_background_noise_":
            continue
        audio_paths = GetAudioPaths(root_dir, command, seed, percentage)
        if command in pretrain_dict:
            labels = [pretrain_dict[command]] * len(audio_paths)
            pretrain_train += zip(audio_paths[:train_files], labels[:train_files])
            pretrain_eval += zip(audio_paths[train_files:train_files + valid_files], labels[train_files:train_files + valid_files])
            pretrain_test += zip(audio_paths[train_files + valid_files:], labels[train_files + valid_files:])
        elif command in finetune_dict:
            labels = [finetune_dict[command]] * len(audio_paths)
            finetune_train += zip(audio_paths[:train_files], labels[:train_files])
            finetune_eval += zip(audio_paths[train_files:train_files + valid_files], labels[train_files:train_files + valid_files])
            finetune_test += zip(audio_paths[train_files + valid_files:], labels[train_files + valid_files:])
        else:
            labels = [1] * len(audio_paths)
            pretrain_train += zip(audio_paths[:train_files], labels[:train_files])
            pretrain_eval += zip(audio_paths[train_files:train_files + valid_files], labels[train_files:train_files + valid_files])
            pretrain_test += zip(audio_paths[train_files + valid_files:], labels[train_files + valid_files:])
            finetune_train += zip(audio_paths[:train_files], labels[:train_files])
            finetune_eval += zip(audio_paths[train_files:train_files + valid_files], labels[train_files:train_files + valid_files])
            finetune_test += zip(audio_paths[train_files + valid_files:], labels[train_files + valid_files:])
    
    # add silence using the same amount of files as the other commands
    # use make_empty_audio to create the silence file
    make_empty_audio(root_dir)
    for _ in range(train_files):
        pretrain_train.append((os.path.join(root_dir, "_silence_.wav"), 0))
        finetune_train.append((os.path.join(root_dir, "_silence_.wav"), 0))
    for _ in range(valid_files):
        pretrain_eval.append((os.path.join(root_dir, "_silence_.wav"), 0))
        finetune_eval.append((os.path.join(root_dir, "_silence_.wav"), 0))
    for _ in range(test_files):
        pretrain_test.append((os.path.join(root_dir, "_silence_.wav"), 0))
        finetune_test.append((os.path.join(root_dir, "_silence_.wav"), 0))
    pretrain_data = {"train": pretrain_train, "valid": pretrain_eval, "test": pretrain_test}
    finetune_data = {"train": finetune_train, "valid": finetune_eval, "test": finetune_test}
    return pretrain_data, pretrain_dict, finetune_data, finetune_dict

class SpeechCommand(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data_list, self.labels = zip(*data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        sample, _ = torchaudio.load(os.path.abspath(audio_path))
        if self.transform:
            sample = self.transform(sample)
        label = self.labels[idx]
        return sample, label

def spec_augment(
    x, frequency_masking_para=20, time_masking_para=20, frequency_mask_num=2, time_mask_num=2
):
    '''
    Apply SpecAugment to the input tensor
    '''
    lenF, lenT = x.shape[1:3]
    # Frequency masking
    for _ in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, lenF - f)
        x[:, f0 : f0 + f, :] = 0
    # Time masking
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, lenT - t)
        x[:, :, t0 : t0 + t] = 0
    return x


class   Preprocess:
    def __init__(
        self,
        noise_loc,
        device,
        hop_length=160,
        win_length=480,
        n_fft=512,
        n_mels=40,
        specaug=False,
        sample_rate=SR,
        frequency_masking_para=7,
        time_masking_para=20,
        frequency_mask_num=2,
        time_mask_num=2,
    ):
        if noise_loc is None:
            self.background_noise = []
        else:
            self.background_noise = [
                torchaudio.load(file_name)[0] for file_name in glob(str(noise_loc) + "/*.wav")
            ]
            assert len(self.background_noise) != 0
        self.feature = LogMel(
            device,
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.sample_len = sample_rate
        self.specaug = specaug
        self.device = device
        if self.specaug:
            self.frequency_masking_para = frequency_masking_para
            self.time_masking_para = time_masking_para
            self.frequency_mask_num = frequency_mask_num
            self.time_mask_num = time_mask_num
            print(
                "frequency specaug %d %d" % (self.frequency_mask_num, self.frequency_masking_para)
            )
            print("time specaug %d %d" % (self.time_mask_num, self.time_masking_para))


    def __call__(self, x, labels, augment=True, noise_prob=0.8, is_train=True):
        assert len(x.shape) == 3
        if augment:
            for idx in range(x.shape[0]):
                if labels[idx] != 0 and (not is_train or random.random() > noise_prob):
                    continue
                noise_amp = (
                    np.random.uniform(0, 0.1) if labels[idx] != 0 else np.random.uniform(0, 1)
                )
                noise = random.choice(self.background_noise).to(self.device)
                sample_loc = random.randint(0, noise.shape[-1] - self.sample_len)
                noise = noise_amp * noise[:, sample_loc : sample_loc + SR]

                if is_train:
                    x_shift = int(np.random.uniform(-0.1, 0.1) * SR)
                    zero_padding = torch.zeros(1, np.abs(x_shift)).to(self.device)
                    if x_shift < 0:
                        temp_x = torch.cat([zero_padding, x[idx, :, :x_shift]], dim=-1)
                    else:
                        temp_x = torch.cat([x[idx, :, x_shift:], zero_padding], dim=-1)
                    x[idx] = temp_x + noise
                else:  # valid
                    x[idx] = x[idx] + noise
                x[idx] = torch.clamp(x[idx], -1.0, 1.0)

        x = self.feature(x)
        if self.specaug:
            for i in range(x.shape[0]):
                x[i] = spec_augment(
                    x[i],
                    self.frequency_masking_para,
                    self.time_masking_para,
                    self.frequency_mask_num,
                    self.time_mask_num,
                )
        return x


class LogMel:
    def __init__(
        self, device, sample_rate=SR, hop_length=160, win_length=480, n_fft=512, n_mels=40
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.device = device


    def __call__(self, x):
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output


class Padding:
    """zero pad to have 1 sec len"""

    def __init__(self):
        self.output_len = SR

    def __call__(self, x):
        pad_len = self.output_len - x.shape[-1]
        if pad_len > 0:
            x = torch.cat([x, torch.zeros([x.shape[0], pad_len])], dim=-1)
        elif pad_len < 0:
            raise ValueError("no sample exceed 1sec in GSC.")
        return x
    

def DownloadDataset(loc, url):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576
    with open(os.path.join(loc, filename), "wb") as f:
        for data in response.iter_content(block_size):
            f.write(data)
            read_so_far = f.tell()
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                print(f"Downloaded {read_so_far} of {total_size} bytes ({percent:.2f}%)")
    with tarfile.open(os.path.join(loc, filename), "r:gz") as tar:
        tar.extractall(loc)


def make_empty_audio(loc):
    '''
    Create an empty audio file for _silence_
    '''
    if not os.path.isfile(loc / "_silence_.wav"):
        zeros = torch.zeros([1, SR])
        torchaudio.save(loc / "_silence_.wav", zeros, SR)

def GenerateConfusionMatrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    return confusion_matrix

def VisualizeConfusionMatrix(y_true, y_pred, class_dict=label_dict, save_path=None):
    labels = class_dict.keys()
    confusion_matrix = GenerateConfusionMatrix(y_true, y_pred, len(labels))
    
    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.matshow(confusion_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    tick_positions = range(len(labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(list(labels), rotation=90)
    ax.set_yticklabels(list(labels)) 

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(Path(save_path))