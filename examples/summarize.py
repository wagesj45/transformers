import argparse
from collections import namedtuple
import functools
import logging
import sys

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from model_bertabs import BertAbsSummarizer, TransformerDecoderState
from transformers.generate import BeamSearch
from transformers import BertTokenizer

from utils_summarization import (
    CNNDailyMailDataset,
    encode_for_summarization,
    build_mask,
    fit_to_block_size,
    compute_token_type_ids,
    build_lm_labels,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_and_cache_examples(args, tokenizer):
    dataset = CNNDailyMailDataset(args.data_dir)
    return dataset


def collate(data, tokenizer, block_size):
    """ List of tuple as an input. """
    # remove the files with empty an story/summary, encode and fit to block
    data = filter(lambda x: not (len(x[0]) == 0 or len(x[1]) == 0), data)
    data = [encode_for_summarization(story, summary, tokenizer) for story, summary in data]
    data = [
        (
            fit_to_block_size(story, block_size, tokenizer.pad_token_id),
            fit_to_block_size(summary, block_size, tokenizer.pad_token_id),
        )
        for story, summary in data
    ]

    stories = torch.tensor([story for story, summary in data])
    summaries = torch.tensor([summary for story, summary in data])
    encoder_token_type_ids = compute_token_type_ids(stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(stories, tokenizer.pad_token_id)
    decoder_mask = build_mask(summaries, tokenizer.pad_token_id)
    lm_labels = build_lm_labels(summaries, tokenizer.pad_token_id)

    return (
        stories,
        summaries,
        encoder_token_type_ids,
        encoder_mask,
        decoder_mask,
        lm_labels,
    )


# --------------
# Load the model
# --------------

def get_pretrained_BertAbs_model(path):
    BertAbsConfig = namedtuple(
        "BertAbsConfig",
        ["temp_dir", "large", "finetune_bert", "encoder", "share_emb", "max_pos", "enc_layers", "enc_hidden_size", "enc_heads", "enc_ff_size", "enc_dropout", "dec_layers", "dec_hidden_size", "dec_heads", "dec_ff_size", "dec_dropout"],
    )

    config = BertAbsConfig(
        temp_dir=".",
        finetune_bert=False,
        large=False,
        share_emb=True,
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
    checkpoints = torch.load(path, lambda storage, loc: storage)
    bertabs = BertAbsSummarizer.from_pretrained(checkpoints, config, torch.device("cpu"))
    bertabs.eval()
    return bertabs


# -------------
# Summarization
# -------------

def evaluate(args, model, tokenizer):
    eval_dataset = load_and_cache_examples(args, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_collate_fn = functools.partial(collate, tokenizer=tokenizer, block_size=512)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_collate_fn,
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        source, _, encoder_token_type_ids, encoder_mask, _, _ = batch
        source = source.to(args.device)
        encoder_token_type_ids = encoder_token_type_ids.to(args.device)
        encoder_mask = encoder_mask.to(args.device)
        summaries_tokens = summarize(
            args, source, encoder_token_type_ids, encoder_mask, model, tokenizer, args.device
        )
        for summary_tokens in summaries_tokens:
            sentences = decode_summary(summary_tokens, tokenizer)
            print(sentences)


def summarize(args, source, encoder_token_type_ids, encoder_mask, model, tokenizer, device):
    """ Summarize a whole batch returned by the data loader.
    """

    model_kwargs = {
        "encoder_token_type_ids": encoder_token_type_ids,
        # "encoder_attention_mask": encoder_mask,
        "decoder_state": TransformerDecoderState(source),
    }

    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'], 'PAD': tokenizer.vocab['[PAD]']}

    batch_size = source.size(0)
    with torch.no_grad():
        beam = BeamSearch(
            model,
            symbols['BOS'],
            symbols['PAD'],
            symbols['EOS'],
            batch_size=batch_size,
            beam_size=5,
            min_length=5,
            max_length=10,
            alpha=0.9,
            block_repeating_trigrams=True,
            device=device,
        )

        results = beam(source, **model_kwargs)

    best_predictions_idx = [
        max(enumerate(results["scores"][i]), key=lambda x: x[1])[0]
        for i in range(batch_size)
    ]
    summaries_tokens = [
        results["predictions"][b][idx]
        for b, idx in zip(range(batch_size), best_predictions_idx)
    ]

    return summaries_tokens


def decode_summary(summary_tokens, tokenizer):
    """ Decode the summary and return it in a format
    suitable for evaluation.
    """
    summary_tokens = summary_tokens.to("cpu").numpy()
    summary = tokenizer.decode(summary_tokens)
    sentences = summary.split(".")
    sentences = [s + "." for s in sentences]
    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    args = parser.parse_args()
    args.device = torch.device("cpu")

    # Model & Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = get_pretrained_BertAbs_model('bert-ext-abs.pt')

    # Evaluate
    evaluate(args, model, tokenizer)
