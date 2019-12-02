import warnings

import torch
import torch.nn.functional as F
from tqdm import trange


class Sampler(object):
    r""" Sampler is used to generate sequences of ids from logit inputs.

    Attributes:
        **config**: ``SamplerConfig``
            Configuration of the sampler which includes the following variables
                - k: parameter for the top-k filtering
                - p: parameter for the nucleus filtering
                - temperature: parameter used to modulate the distribution over ids
                - repetition_penalty: the penalty that repeating ids incur
        **device**: ``torch.device``
            Device on which the computations will be run.
    """

    def __init__(self, device, k=0, p=0, do_sample=True, temperature=1., repetition_penalty=1.):
        self.k = k
        self.p = p
        self.do_sample = do_sample
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

        self.do_apply_repetition_penalty = True if repetition_penalty > 1 else False

        if self.p > 1:
            warnings.warn(
                """You are trying to apply nucleus filtering with a value of p greater than 1 ({}).
                However p is a probability and its value must lie between 0 and 1. In effect, no filtering
                will be applied. If this is not the behavior you expect, change the value of p.""".format(
                    self.p
                )
            )

        self.device = device

    def get_one_token(self, next_token_logits, past_sequence):
        logits = self.apply_repetition_penalty(next_token_logits, past_sequence)
        if self.do_sample:
            logits = self.apply_temperature(logits)
            logits = self.apply_top_k_filter(logits)
            logits = self.apply_nucleus_filter(logits)
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return torch.argmax(logits, dim=-1).unsqueeze(-1)

    def apply_repetition_penalty(self, logits, past_sequence):
        """ Apply a penalty to tokens that appear more than once in the
        generated sequence.

        .. Keskar, Nitish Shirish, et al. "Ctrl: A conditional transformer
           language model for controllable generation." arXiv preprint
           arXiv:1909.05858 (2019).
        """
        if self.do_apply_repetition_penalty:
            generated_token_idx = set(past_sequence[0].tolist())
            for token_idx in generated_token_idx:
                logits[0, token_idx] /= self.repetition_penalty
        return logits

    def apply_temperature(self, logits):
        """ Shape the tokens' distribution through temperature. The higher the value
        of the temperature, the more skewed towards high probability events the
        distribution is.

        .. Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning.
        MIT press, 2016.
        """
        # when dividing a float by 0, torch returns inf which in turns breaks the
        # multinomial with an error message that is not very helpful. It is better
        # for the user to break the execution and explain why.
        if self.temperature == 0:
            raise ZeroDivisionError(
                """You are trying to sample with a temperature equal to 0.
                If you wanted to do greedy sampling, set instead `do_sample` to False.
                Otherwise set the temperature to a value different from 0."""
            )
        return logits / self.temperature

    def apply_top_k_filter(self, logits):
        """ Use the probability distribution of the tokens to determine the set
        to be sampled from. Specifically we select the set of size k such that
        the sum of its items' probabilities is maximum.

        .. Fan, Angela, Mike Lewis, and Yann Dauphin. "Hierarchical neural
        story generation." arXiv preprint arXiv:1805.04833 (2018).
        """
        if self.k > 0:
            vocabulary_size = logits.size(-1)
            if self.k > vocabulary_size:
                warnings.warn(
                    """You provided a value for k ({}) that is larger than the vocabulary size ({}).
                    We adjusted k's value to the vocabulary size; if that was what you intended to do
                    we recommend setting k to 0 instead. It this is not the behavior you expected,
                    choose a value of k that is smaller than the vocabulary size.""".format(
                        self.k, vocabulary_size
                    )
                )
                self.k = vocabulary_size

            indices_to_remove = logits < torch.topk(logits, self.k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        return logits

    def apply_nucleus_filter(self, logits):
        """ Use the probability distribution of the tokens to determine the set
        to be sampled from. Specifically, choose the smallest set such that the
        sum of its items' probabilities is greater than a number p in [0,1].

        .. Holtzman, Ari, et al. "The curious case of neural text
           degeneration." arXiv preprint arXiv:1904.09751 (2019).
        """
        if self.p > 0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probabilities = F.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

            # Remove tokens with cumulative probability above the threshold,
            # but keep the first token above the threshold.
            sorted_indices_to_remove = cumulative_probabilities > self.p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        return logits


class SamplerEncoderDecoder(Sampler):
    """ Generic sampler for encoder-decoder models.
    """

    def __init__(self, model, config, device):
        self.model = model
        super(SamplerEncoderDecoder, self).__init__(config, device)

    def generate_sequence(self, encoder_input_ids, prompt_ids, length=1, **model_kwargs):
        encoder_kwargs, decoder_kwargs = self.model.prepare_model_kwargs(**model_kwargs)
        with torch.no_grad():
            encoder_outputs = self.model.encode(encoder_input_ids, **encoder_kwargs)
            encoder_hidden_states = encoder_outputs[0]

            generated_sequence = prompt_ids
            for _ in trange(length):
                outputs = self.model.decode(
                    generated_sequence,
                    encoder_hidden_states=encoder_hidden_states,
                    **decoder_kwargs
                )
                next_tokens_logits = outputs[0][:, -1, :]
                next_tokens = self.get_one_token(
                    next_tokens_logits, generated_sequence
                )
                generated_sequence = torch.cat((generated_sequence, next_tokens), dim=1)

        return generated_sequence.squeeze(0).tolist()
