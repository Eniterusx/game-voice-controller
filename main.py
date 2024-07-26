# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from os import path
from argparse import ArgumentParser
import shutil
from glob import glob
from pathlib import Path

from timeit import default_timer as timer
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, VisualizeConfusionMatrix, SplitDataset, FineTuneSplit

class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        parser.add_argument("--output", help="output folder", default="models/output", type=str)
        parser.add_argument("--fine_tune", help="fine tune option", default=False, action="store_true")
        parser.add_argument("--percentage", help="percentage of data to use", default=100, type=int)
        parser.add_argument("--seed", help="random seed", default=42, type=int)
        parser.add_argument("--epochs", help="number of epochs", default=100, type=int)
        parser.add_argument("--config", help="yaml config file (overwrites other hiperparam settings)", default=None, type=str)
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._set_hyperparameters()
        os.makedirs(self.output, exist_ok=True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.fine_tune:
            self._load_finetune_data()
        else:
            self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch = self.epochs
        warmup_epoch = self.warmup_epoch
        init_lr = self.init_lr
        lr_lower_limit = self.lr_lower_limit

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=self.weight_decay, momentum=self.momentum)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        best_acc = 0.0
        self.best_model = None
        # train
        timer_ = timer()
        for epoch in range(total_epoch):
            self.model.train()

            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))
                if valid_acc >= best_acc:
                    best_acc = valid_acc
                    print(f"New best accuracy: {best_acc:.3f}")

        test_acc, precission, recall, f1_score = self.Test(self.test_dataset, self.test_loader, augment=False, visualize=True, calc_confusion_matrix=True)  # official testset
        print("test acc: %.3f" % (test_acc))
        print(f"Total time: {timer() - timer_}")
        print("Best acc: %.3f" % best_acc)
        # save best acc as a txt file
        with open(path.join(self.output, "best_acc.txt"), "w") as f:
            f.write(f"Best acc: {best_acc:.3f}\n")
            f.write(f"Test acc: {test_acc:.3f}\n")
            f.write(f"Precision: {precission:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"F1 score: {f1_score:.3f}\n")
            f.write(f"Train commands: {list(self.label_dict.keys())}\n")
        torch.save(self.model.state_dict(), path.join(self.output, "model_best.pth"))
        if self.fine_tune:
            self.FineTune()
        print("End.")
    
    def FineTune(self):
        # freeze all layers except the last fc layer
        # change the last fc layer to have as many output classes as self.finetune_dict
        self._fine_tune()

        # train hyperparameters
        total_epoch = self.fine_tune_epochs
        warmup_epoch = self.warmup_epoch
        init_lr = self.init_lr
        lr_lower_limit = self.lr_lower_limit

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=self.weight_decay, momentum=self.momentum)
        n_step_warmup = len(self.finetune_train_loader) * warmup_epoch
        total_iter = len(self.finetune_train_loader) * total_epoch
        iterations = 0

        best_acc = 0.0
        # train
        timer_ = timer()
        for epoch in range(total_epoch):
            self.model.train()

            for sample in tqdm(self.finetune_train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.finetune_valid_dataset, self.finetune_valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))
                if valid_acc >= best_acc:
                    best_acc = valid_acc
                    print(f"New best accuracy: {best_acc:.3f}")
    
        test_acc, precision, recall, f1_score = self.Test(self.finetune_test_dataset, self.finetune_test_loader, augment=False, visualize=True, finetune=True, calc_confusion_matrix=True)  # official testset
        print("test acc: %.3f" % (test_acc))
        print(f"Total finetune time: {timer() - timer_}")
        print("Best acc: %.3f" % best_acc)
        # save best acc as a txt file
        with open(path.join(self.output, "best_acc_finetune.txt"), "w") as f:
            f.write(f"Best acc: {best_acc:.3f}\n")
            f.write(f"Test acc: {test_acc:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"F1 score: {f1_score:.3f}\n")
            f.write(f"Fine tune commands: {list(self.finetune_dict.keys())}\n")
        torch.save(self.model.state_dict(), path.join(self.output, "model_best_finetune.pth"))
        print("End of fine tuning.")

    def Test(self, dataset, loader, augment, visualize=False, finetune=False, calc_confusion_matrix=False):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        print("Testing...")
        if visualize:
            predicted_labels = []
            true_labels = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
            if calc_confusion_matrix:
                for predict, label in zip(prediction, labels):
                    if predict == label:
                        if predict in [0, 1]:
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if predict in [0, 1]:
                            FN += 1
                        else:
                            FP += 1
            
            if visualize:
                predicted_labels.extend(prediction.detach().cpu().numpy())
                true_labels.extend(labels.detach().cpu().numpy())
        
        if visualize:
            if finetune:
                output = path.join(self.output, "confusion_matrix_finetune.png")
                cmd_dict = self.finetune_dict
            else:
                output = path.join(self.output, "confusion_matrix.png")
                cmd_dict = self.label_dict
            VisualizeConfusionMatrix(true_labels, predicted_labels, class_dict=cmd_dict, save_path=output)
        acc = true_count / num_testdata * 100.0  # percentage
        if not calc_confusion_matrix:
            return acc
        else:
            print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
            precision = TP / max((TP + FP), 1)
            recall = TP / max((TP + FN), 1)
            f1_score = 2 * TP / max((2 * TP + FP + FN), 1)
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1 score: {f1_score:.3f}")
            return acc, precision, recall, f1_score 

    def _set_hyperparameters(self):
        if self.config:
            with open(self.config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(config)
            assert self.init_lr > 0, "init_lr must be a positive float"
            assert self.lr_lower_limit >= 0, "lr_lower_limit must be a non-negative float"
            assert self.warmup_epoch > 0, "warmup_epoch must be a positive integer"
            assert self.weight_decay >= 0, "weight_decay must be a non-negative float"
            assert self.momentum >= 0, "momentum must be a non-negative float"
            assert self.batch_size > 0, "batch_size must be a positive integer"
        else:
            self.init_lr = 1e-1
            self.lr_lower_limit = 0
            self.warmup_epoch = 5
            self.weight_decay = 1e-3
            self.momentum = 0.9
            self.batch_size = 100
            self.seed = 42
            self.fine_tune_epochs = self.epochs

        self.percentage = self.percentage / 100

        assert self.ver in [1, 2], "ver must be 1 or 2"
        assert self.tau in [1, 1.5, 2, 3, 6, 8], "tau must be 1, 1.5, 2, 3, 6, or 8"
        assert self.gpu >= 0, "gpu must be a non-negative integer"
        assert self.percentage > 0.0 and self.percentage <= 1.0, "percentage must be between 1 and 100"
        assert self.seed > 0, "seed must be a positive integer"
        assert self.epochs > 0, "epochs must be a positive integer"

    def _check_dataset(self):
        """
        Private method that downloads the data if necessary.
        """
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            print("Done...")

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        self._check_dataset()

        base_dir = "./data/speech_commands_v0.01"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
        base_dir = Path(base_dir)
        noise_dir = base_dir / "_background_noise_"

        transform = transforms.Compose([Padding()])

        train_data, self.label_dict = SplitDataset(base_dir, self.seed, self.percentage)
        train_train, train_valid, train_test = train_data["train"], train_data["valid"], train_data["test"]

        self.train_dataset = SpeechCommand(train_train, transform)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = SpeechCommand(train_valid, transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=0)
        self.test_dataset = SpeechCommand(train_test, transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

        print(
            "check num of data in train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        print(
            "check noise folder num of data %d"
            % len(list(noise_dir.glob("*.wav")))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)   

    def _load_finetune_data(self):
        """
        Private method that loads data into the object.
        Used for fine-tuning the model.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        self._check_dataset()

        base_dir = "./data/speech_commands_v0.01"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
        base_dir = Path(base_dir)
        noise_dir = base_dir / "_background_noise_"

        transform = transforms.Compose([Padding()])

        pretrain_data, self.label_dict, finetune_data, self.finetune_dict = FineTuneSplit(base_dir, self.seed, self.percentage)
        pretrain_train, pretrain_valid, pretrain_test = pretrain_data["train"], pretrain_data["valid"], pretrain_data["test"]
        finetune_train, finetune_valid, finetune_test = finetune_data["train"], finetune_data["valid"], finetune_data["test"]

        self.train_dataset = SpeechCommand(pretrain_train, transform)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = SpeechCommand(pretrain_valid, transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=0)
        self.test_dataset = SpeechCommand(pretrain_test, transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

        self.finetune_train_dataset = SpeechCommand(finetune_train, transform)
        self.finetune_train_loader = DataLoader(
            self.finetune_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        self.finetune_valid_dataset = SpeechCommand(finetune_valid, transform)
        self.finetune_valid_loader = DataLoader(self.finetune_valid_dataset, batch_size=self.batch_size, num_workers=0)
        self.finetune_test_dataset = SpeechCommand(finetune_test, transform)
        self.finetune_test_loader = DataLoader(self.finetune_test_dataset, batch_size=self.batch_size, num_workers=0)

        print(
            "check num of data in pretrain train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        print(
            "check num of data in finetune train/valid/test %d/%d/%d"
            % (len(self.finetune_train_dataset), len(self.finetune_valid_dataset), len(self.finetune_test_dataset))
        )

        print(
            "check noise folder num of data %d"
            % len(list(noise_dir.glob("*.wav")))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _fine_tune(self):
        # freeze all layers except the last fc layer
        for param in self.model.parameters():
            param.requires_grad = False

        # change the last fc layer to have as many output classes as self.finetune_dict
        self.model.rebuild_classifier(len(self.finetune_dict))

        for param in self.model.classifier.parameters():
            param.requires_grad = True
        self.model = self.model.to(self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)


if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()
