# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
# limitations under the License.

from torch import nn


def get_activation(act_fn):
    if act_fn in ["swish", "silu"]:
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")