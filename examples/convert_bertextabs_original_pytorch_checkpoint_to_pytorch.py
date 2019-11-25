# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Convert BertExtAbs's checkpoints """

import argparse
from collections import namedtuple
import logging

import torch

from models.model_builder import AbsSummarizer  # The authors' implementation
from model_bertabs import BertAbsSummarizer

from transformers import BertTokenizer, BertConfig, Model2Model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_TEXT = 'Hello world! cécé herlolip'


BertAbsConfig = namedtuple(
    "BertAbsConfig",
    ["temp_dir", "large", "use_bert_emb", "finetune_bert", "encoder", "share_emb", "max_pos", "enc_layers", "enc_hidden_size", "enc_heads", "enc_ff_size", "enc_dropout", "dec_layers", "dec_hidden_size", "dec_heads", "dec_ff_size", "dec_dropout"],
)


def convert_bertabs_checkpoints(path_to_checkpoints, dump_path):
    """ Copy/paste and tweak the pre-trained weights provided by the creators
    of BertAbs for the internal architecture.
    """

    # Instantiate the authors' model with the pre-trained weights
    config = BertAbsConfig(
        temp_dir=".",
        finetune_bert=False,
        large=False,
        share_emb=True,
        use_bert_emb=False,
        encoder="bert",
        max_pos=512,
        enc_layers=6,
        enc_hidden_size=512,
        enc_heads=8,
        enc_ff_size=512,
        enc_dropout=0.2,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        dec_dropout=0.2,
    )
    checkpoints = torch.load(path_to_checkpoints, lambda storage, loc: storage)
    original = AbsSummarizer(config, torch.device("cpu"), checkpoints)
    original.eval()

    new_model = BertAbsSummarizer(config, torch.device("cpu"))

    # Convert the weights
    new_model.encoder.load_state_dict(original.bert.state_dict())
    new_model.decoder.load_state_dict(original.decoder.state_dict())
    new_model.generator.load_state_dict(original.generator.state_dict())

    # The model has been saved with torch.save(model) and this is bound to the exact
    # directory structure. We save the state_dict instead.
    torch.save(new_model.state_dict(), "bert-ext-abs.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bertabs_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch dump.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()

    convert_bertabs_checkpoints(
        args.bertabs_checkpoint_path,
        args.pytorch_dump_folder_path,
    )
