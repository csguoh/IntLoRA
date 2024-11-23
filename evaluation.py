#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import logging
import math
import os
import warnings
from pathlib import Path

from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel

import lpips
import json
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torchvision.transforms.functional as TF
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage



def get_prompt(subject_name, prompt_idx):
    subject_names = [
        "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
        "candle", "cat", "cat2", "clock", "colorful_sneaker",
        "dog", "dog2", "dog3", "dog5", "dog6",
        "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie",
        "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
        "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie",
    ]

    class_tokens = [
        "backpack", "backpack", "stuffed animal", "bowl", "can",
        "candle", "cat", "cat", "clock", "sneaker",
        "dog", "dog", "dog", "dog", "dog",
        "dog", "dog", "toy", "boot", "stuffed animal",
        "toy", "glasses", "toy", "toy", "cartoon",
        "toy", "sneaker", "teapot", "vase", "stuffed animal",
    ]

    class_token = class_tokens[subject_names.index(subject_name)]

    prompt_list = [
        f"a qwe {class_token} in the jungle",
        f"a qwe {class_token} in the snow",
        f"a qwe {class_token} on the beach",
        f"a qwe {class_token} on a cobblestone street",
        f"a qwe {class_token} on top of pink fabric",
        f"a qwe {class_token} on top of a wooden floor",
        f"a qwe {class_token} with a city in the background",
        f"a qwe {class_token} with a mountain in the background",
        f"a qwe {class_token} with a blue house in the background",
        f"a qwe {class_token} on top of a purple rug in a forest",
        f"a qwe {class_token} wearing a red hat",
        f"a qwe {class_token} wearing a santa hat",
        f"a qwe {class_token} wearing a rainbow scarf",
        f"a qwe {class_token} wearing a black top hat and a monocle",
        f"a qwe {class_token} in a chef outfit",
        f"a qwe {class_token} in a firefighter outfit",
        f"a qwe {class_token} in a police outfit",
        f"a qwe {class_token} wearing pink glasses",
        f"a qwe {class_token} wearing a yellow shirt",
        f"a qwe {class_token} in a purple wizard outfit",
        f"a red qwe {class_token}",
        f"a purple qwe {class_token}",
        f"a shiny qwe {class_token}",
        f"a wet qwe {class_token}",
        f"a cube shaped qwe {class_token}",
    ]

    return prompt_list[int(prompt_idx)]


class PromptDatasetCLIP(Dataset):
    def __init__(self, subject_name, data_dir_B, tokenizer, processor, epoch=None):
        self.data_dir_B = data_dir_B

        subject_name, prompt_idx = subject_name.split('-')

        data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        self.image_lst = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]
        self.prompt_lst = [get_prompt(subject_name, prompt_idx)] * len(self.image_lst)

        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs


class PairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject_name, data_dir_A, data_dir_B, processor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B

        subject_name, prompt_idx = subject_name.split('-')

        self.data_dir_A = os.path.join(self.data_dir_A, subject_name)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if
                              f.endswith(".jpg")]

        data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        self.image_files_B = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)

        image_A = Image.open(self.image_files_A[index_A])  # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B])  # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(
                min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject_name, data_dir_A, data_dir_B, feature_extractor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B

        subject_name, prompt_idx = subject_name.split('-')

        self.data_dir_A = os.path.join(self.data_dir_A, subject_name)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if
                              f.endswith(".jpg")]

        data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        self.image_files_B = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)

        image_A = Image.open(self.image_files_A[index_A])  # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B])  # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(
                min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject_name, data_dir_A, data_dir_B, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B

        subject_name, prompt_idx = subject_name.split('-')

        self.data_dir_A = os.path.join(self.data_dir_A, subject_name)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if
                              f.endswith(".jpg")]

        data_dir_B = os.path.join(self.data_dir_B, str(epoch))
        self.image_files_B = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if f.endswith(".png")]

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)

        image_A = Image.open(self.image_files_A[index_A])  # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B])  # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(
                min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_text_model =  CLIPModel.from_pretrained("/data2/xxx/pretrained/clip-vit-large-patch14").to(device)
clip_text_tokenizer = AutoTokenizer.from_pretrained("/data2/xxx/pretrained/clip-vit-large-patch14")
clip_text_processor =  AutoProcessor.from_pretrained("/data2/xxx/pretrained/clip-vit-large-patch14")

def clip_text(subject_name, image_dir):
    criterion = 'clip_text'

    model = clip_text_model
    # Get the text features
    tokenizer = clip_text_tokenizer
    # Get the image features
    processor =clip_text_processor

    epochs = sorted([int(epoch) for epoch in os.listdir(image_dir)])
    best_mean_similarity = 0
    mean_similarity_list = []
    for epoch in epochs:
        similarity = []
        dataset = PromptDatasetCLIP(subject_name, image_dir, tokenizer, processor, epoch)
        dataloader = DataLoader(dataset, batch_size=32)
        for i in range(len(dataset)):
            image_inputs, prompt_inputs = dataset[i]
            if image_inputs is not None and prompt_inputs is not None:
                image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
                prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
                prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
                # print(prompt_inputs)
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**prompt_inputs)

                sim = cosine_similarity(image_features, text_features)

                # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                # text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                # logit_scale = model.logit_scale.exp()
                # sim = torch.matmul(text_features, image_features.t()) * logit_scale
                similarity.append(sim.item())

        if similarity:
            mean_similarity = torch.tensor(similarity).mean().item()
            mean_similarity_list.append(mean_similarity)
            best_mean_similarity = max(best_mean_similarity, mean_similarity)
            print(f'epoch: {epoch}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')
        else:
            mean_similarity_list.append(0)
            print(f'epoch: {epoch}, criterion: {criterion}, mean_similarity: {0}({best_mean_similarity})')

    return mean_similarity_list


clip_image_model = CLIPModel.from_pretrained("/data2/xxx/pretrained/clip-vit-large-patch14").to(device)
clip_image_processor = AutoProcessor.from_pretrained("/data2/xxx/pretrained/clip-vit-large-patch14")

def clip_image(subject_name, image_dir, dreambooth_dir='/data2/xxx/dataset/dreambooth'):
    criterion = 'clip_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = clip_image_model
    # Get the image features
    processor = clip_image_processor

    epochs = sorted([int(epoch) for epoch in os.listdir(image_dir)])
    best_mean_similarity = 0
    mean_similarity_list = []
    for epoch in epochs:
        similarity = []
        dataset = PairwiseImageDatasetCLIP(subject_name, dreambooth_dir, image_dir, processor, epoch)
        # dataset = SelfPairwiseImageDatasetCLIP(subject, './data', processor)

        for i in range(len(dataset)):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device)

                image_A_features = model.get_image_features(**inputs_A)
                image_B_features = model.get_image_features(**inputs_B)

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                sim = torch.matmul(image_A_features, image_B_features.t())  # * logit_scale
                similarity.append(sim.item())

        if similarity:
            mean_similarity = torch.tensor(similarity).mean().item()
            best_mean_similarity = max(best_mean_similarity, mean_similarity)
            mean_similarity_list.append(mean_similarity)
            print(f'epoch: {epoch}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')
        else:
            mean_similarity_list.append(0)
            print(f'epoch: {epoch}, criterion: {criterion}, mean_similarity: {0}({best_mean_similarity})')

    return mean_similarity_list


dino_model = ViTModel.from_pretrained('/data2/xxx/pretrained/dino-vits16').to(device)
deno_feature_extractor =  ViTFeatureExtractor.from_pretrained('/data2/xxx/pretrained/dino-vits16')

def dino(subject_name, image_dir, dreambooth_dir='/data2/xxx/dataset/dreambooth'):
    criterion = 'dino'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dino_model
    feature_extractor = deno_feature_extractor

    epochs = sorted([int(epoch) for epoch in os.listdir(image_dir)])
    best_mean_similarity = 0
    mean_similarity_list = []
    for epoch in epochs:
        similarity = []
        # dataset = PairwiseImageDatasetDINO(subject, './data', image_dir, feature_extractor, epoch)
        dataset = PairwiseImageDatasetDINO(subject_name, dreambooth_dir, image_dir, feature_extractor, epoch)
        # dataset = SelfPairwiseImageDatasetDINO(subject, './data', feature_extractor)

        for i in range(len(dataset)):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device)

                outputs_A = model(**inputs_A)
                image_A_features = outputs_A.last_hidden_state[:, 0, :]

                outputs_B = model(**inputs_B)
                image_B_features = outputs_B.last_hidden_state[:, 0, :]

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                sim = torch.matmul(image_A_features, image_B_features.t())  # * logit_scale
                similarity.append(sim.item())

        mean_similarity = torch.tensor(similarity).mean().item()
        best_mean_similarity = max(best_mean_similarity, mean_similarity)
        mean_similarity_list.append(mean_similarity)
        print(f'epoch: {epoch}, criterion: {criterion}, mean_similarity: {mean_similarity}({best_mean_similarity})')

    return mean_similarity_list


lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

def lpips_image(subject_name, image_dir,dreambooth_dir='/data2/xxx/dataset/dreambooth'):
    criterion = 'lpips_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up the LPIPS model (vgg=True uses the VGG-based model from the paper)

    loss_fn = lpips_loss_fn
    # 有可能有些epoch没跑全
    epochs = sorted([int(epoch) for epoch in os.listdir(image_dir)])
    mean_similarity_list = []
    best_mean_similarity = 0
    for epoch in epochs:
        similarity = []
        dataset = PairwiseImageDatasetLPIPS(subject_name, dreambooth_dir, image_dir, epoch)
        # dataset = SelfPairwiseImageDatasetLPIPS(subject, './data')

        for i in range(len(dataset)):
            image_A, image_B = dataset[i]
            if image_A is not None and image_B is not None:
                image_A = image_A.to(device)
                image_B = image_B.to(device)

                # Calculate LPIPS between the two images
                distance = loss_fn(image_A, image_B)

                similarity.append(distance.item())

        mean_similarity = torch.tensor(similarity).mean().item()
        best_mean_similarity = max(best_mean_similarity, mean_similarity)
        mean_similarity_list.append(mean_similarity)
        print(f'epoch: {epoch}, criterion: LPIPS distance, mean_similarity: {mean_similarity}({best_mean_similarity})')

    return mean_similarity_list


if __name__ == "__main__":
    image_dir = '/data2/xxx/ControlNet/oft-db/log_quant'

    subject_dirs, subject_names = [], []
    for name in os.listdir(image_dir):
        if os.path.isdir(os.path.join(image_dir, name)):
            subject_dirs.append(os.path.join(image_dir, name))
            subject_names.append(name)

    results_path = '/data2/xxx/ControlNet/oft-db/log_quant/results.json'

    results_dict = dict()
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = f.__iter__()
            while True:
                try:
                    result_json = json.loads(next(results))
                    results_dict.update(result_json)

                except StopIteration:
                    print("finish extraction.")
                    break

    for idx in range(len(subject_names)):
        subject_name = subject_names[idx]
        subject_dir = subject_dirs[idx]
        print(f'evaluating {subject_dir}')
        dino_sim = dino(subject_name, subject_dir)
        clip_i_sim = clip_image(subject_name, subject_dir)
        clip_t_sim = clip_text(subject_name, subject_dir)
        lpips_sim = lpips_image(subject_name, subject_dir)

        subject_result = {'DINO': dino_sim, 'CLIP-I': clip_i_sim, 'CLIP-T': clip_t_sim, 'LPIPS': lpips_sim}
        print(subject_result)

        with open(results_path, 'a') as f:
            json_string = json.dumps({subject_name: subject_result})
            f.write(json_string + "\n")
