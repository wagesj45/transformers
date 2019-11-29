from collections import namedtuple
import warnings

import torch
import torch.nn.functional as F
from tqdm import trange

from transformers import PreTrainedEncoderDecoder

SamplerConfig = namedtuple(
    "SamplerConfig", ["do_sample", "temperature", "k", "p", "repetition_penalty"]
)


def new_sampler(
    model,
    do_sample=True,
    temperature=1.0,
    k=0,
    p=0,
    repetition_penalty=1.0,
    device=torch.device("cpu"),
):
    """ Factory function that returns the appropriate sampler with regards
    to the model passed as a parameter.

    Only single stacks are currently supported.
    """

    sampler_config = SamplerConfig(
        do_sample=do_sample,
        temperature=temperature,
        k=k,
        p=p,
        repetition_penalty=repetition_penalty,
    )

    is_encoder_decoder = isinstance(PreTrainedEncoderDecoder)
    has_decoder = callable(getattr(model, "decode", None))
    if is_encoder_decoder:
        return SamplerEncoderDecoder(model, sampler_config, device)
    elif has_decoder:
        return SamplerSingleStack(model, sampler_config, device)
    else:
        raise ValueError(
            "you need to pass a model that at least defines a `decode` method to use the sampler"
        )


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

    def __init__(self, config, device):
        self.k = config.k
        self.p = config.p
        self.do_sample = config.do_sample
        self.temperature = config.temperature
        self.repetition_penalty = config.repetition_penalty

        self.do_apply_repetition_penalty = True if config.repetition_penalty > 1 else False

        if self.p > 1:
            warnings.warn(
                "p is a probability; its value must lie between 0 and 1, got {}. Ignored.".format(
                    self.p
                )
            )

        self.device = device

    def generate_sequence(self):
        """ Generate a sequence of `length` tokens starting from the
        provided `prompt`. This method is model-specific.
        """
        raise NotImplementedError

    def generate_one_token(self, next_token_logits, past_sequence):
        logits = self.apply_repetition_penalty(next_token_logits, past_sequence)
        logits = self.apply_temperature(logits)
        logits = self.apply_top_k_filter(logits)
        logits = self.apply_nucleus_filter(logits)
        return self.get_one_token(logits)

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
        if self.do_sample:
            try:
                return logits / self.temperature
            except ZeroDivisionError:
                raise ZeroDivisionError(
                    "if you want to sample, the temperature needs to be different from 0"
                )
        return logits

    def apply_top_k_filter(self, logits):
        """ Use the probability distribution of the tokens to determine the set
        to be sampled from.  Specifically we select the set of size k such that
        the sum of its items' probabilities is maximum.

        .. Fan, Angela, Mike Lewis, and Yann Dauphin. "Hierarchical neural
        story generation." arXiv preprint arXiv:1805.04833 (2018).
        """
        if self.k > 0:
            try:
                indices_to_remove = logits < torch.topk(logits, self.k)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")
            except RuntimeError:
                vocabulary_size = logits.size(-1)
                raise RuntimeError(
                    "the value of k provided ({}) must be smaller than the size of the vocabulary ({})".format(
                        self.k, vocabulary_size
                    )
                )

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

    def get_one_token(self, logits):
        if self.do_sample:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return torch.argmax(logits, dim=-1).unsqueeze(-1)


class SamplerSingleStack(Sampler):
    """ Generic sampler for single-stack models.
    """

    def __init__(self, model, config, device):
        super(SamplerSingleStack, self).__init__(config, device)
        self.model = model

        # the followings checks that the model is on the same device as the one
        # that is specified. It only works for models that fit on one GPU.
        model_device = next(model.parameters()).device
        if model_device != device:
            warnings.warn(
                "The model is not on the same device as the one you specified. Expected {}, got {}.".format(
                    device, model_device
                )
            )

    def generate_sequence(self, length=1, prompt_ids=[], **model_kwargs):
        prompt_ids = torch.tensor(
            prompt_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        generated_sequence = prompt_ids
        with torch.no_grad():
            for _ in trange(length):
                outputs = self.model.decode(generated_sequence, **model_kwargs)
                next_tokens_logits = outputs[0][:, -1, :]
                next_tokens = self.generate_one_token(
                    next_tokens_logits, generated_sequence
                )
                generated_sequence = torch.cat((generated_sequence, next_tokens), dim=1)

        return generated_sequence.squeeze(0).tolist()


class SamplerEncoderDecoder(Sampler):
    """ Generic sampler for encoder-decoder models.
    """

    def __init__(self, model, config, device):
        self.model = model
        super(SamplerEncoderDecoder, self).__init__(config, device)

    def generate_sequence(self, encoder_input_ids, length=1, prompt_ids=[], **model_kwargs):
        encoder_kwargs, decoder_kwargs = self.model.prepare_model_kwargs(**model_kwargs)
        encoder_hidden_states = self.model.encode(encoder_input_ids, **encoder_kwargs)

        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_sequence = prompt
        with torch.no_grad():
            for _ in trange(length):
                outputs = self.model.decode(
                    generated_sequence,
                    encoder_hidden_states=encoder_hidden_states,
                    **decoder_kwargs
                )
                next_tokens_logits = outputs[0][:, -1, :]
                next_tokens = self.generate_one_token(
                    next_tokens_logits, generated_sequence
                )
                generated_sequence = torch.cat((generated_sequence, next_tokens), dim=1)

        return generated_sequence.squeeze(0).tolist()
