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

import math
import os
from functools import reduce
import numpy as np

import json

if __name__ == "__main__":
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
    else:
        raise NotImplementedError
    total_result = np.zeros(4)
    metric_name_list = ['DINO', 'CLIP-I', 'CLIP-T', 'LPIPS']
    except_list = []
    num_samples = 0
    print(len((results_dict.keys())))
    for subject_name, subject_results in results_dict.items():
        if subject_name in except_list:
            continue
        num_samples += 1
        metric_results_percent = None
        for metric_name, metric_results in subject_results.items():
            metric_results = [0 if np.isnan(r) else r for r in metric_results]
            try:
                metric_results_norm = np.array(metric_results) / (max(metric_results) - min(metric_results))
            except:
                print(subject_name)
            if metric_results_percent is None:
                metric_results_percent = metric_results_norm
            else:
                metric_results_percent += metric_results_norm

        subject_results_max_idx = np.argmax(metric_results_percent)
        for idx, metric_name in enumerate(metric_name_list):
            total_result[idx] += subject_results[metric_name][subject_results_max_idx]
    total_result /= num_samples
    print(f'DINO: {total_result[0]}, CLIP-I: {total_result[1]}, CLIP-T: {total_result[2]}, LPIPS: {total_result[3]}')
