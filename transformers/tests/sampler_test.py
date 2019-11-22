# coding=utf-8
from collections import namedtuple
import unittest

import numpy as np
import pytest

from transformers import is_torch_available

if is_torch_available():
    import torch

    from transformers import (
        generate,
        XLMConfig,
        XLMTokenizer,
        XLMWithLMHeadModel,
        XLNetConfig,
        XLNetLMHeadModel,
        PreTrainedEncoderDecoder,
    )
    from transformers.generate.sampler import (
        SamplerConfig,
        Sampler,
        SamplerSingleStack,
    )
else:
    pytestmark = pytest.mark.skip("Require Torch")


class SamplerFactoryTest(unittest.TestCase):
    ModelStub = namedtuple("ModelStub", [])

    def test_creation_of_xlm_sampler(self):
        model_config = XLMConfig()
        model = XLMWithLMHeadModel(model_config)
        sampler = generate.new_sampler(model)
        self.assertIsInstance(sampler, SamplerSingleStack)

    def test_creation_of_xlnet_sampler(self):
        model_config = XLNetConfig()
        model = XLNetLMHeadModel(model_config)
        sampler = generate.new_sampler(model)
        self.assertIsInstance(sampler, SamplerSingleStack)

    def test_creation_of_single_stack_model(self):
        model = self.ModelStub()
        sampler = generate.new_sampler(model)
        self.assertIsInstance(sampler, SamplerSingleStack)


class SamplerTest(unittest.TestCase):
    def test_nucleus_sampling(self):
        inf = -float("Inf")
        test_cases = (
            {'p': 0, 'logits': torch.tensor([0.3, 0.1, 0.2]), 'expected': torch.tensor([0.3, 0.1, 0.2])},
            {'p': 0.01, 'logits': torch.tensor([0.3, 0.1, 0.2]), 'expected': torch.tensor([0.3, inf, inf])},
            {'p': 1, 'logits': torch.tensor([0.3, 0.1, 0.2]), 'expected': torch.tensor([0.3, 0.1, 0.2])},
            {'p': 0.2, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, inf, inf])},
            {'p': 0.71, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, inf, 0.2])},
            {'p': 0.71, 'logits': torch.tensor([0.1, 0.7, 0.2]), 'expected': torch.tensor([inf, 0.7, 0.2])},
            {'p': 0.71, 'logits': torch.tensor([0.7, 0.2, 0.1]), 'expected': torch.tensor([0.7, 0.2, inf])},
            {'p': 0.91, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, 0.1, 0.2])},
        )
        for case in test_cases:
            config = SamplerConfig(temperature=1., k=0, p=case["p"], repetition_penalty=1.0)
            sampler = Sampler(config, device=torch.device("cpu"))
            filtered_logits = sampler.apply_nucleus_filter(case["logits"])
            np.testing.assert_array_equal(case["expected"].numpy(), filtered_logits.numpy())

    def test_top_k_filter(self):
        inf = -float("Inf")
        test_cases = (
            {'k': 0, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, 0.1, 0.2])},
            {'k': 1, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, inf, inf])},
            {'k': 2, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, inf, 0.2])},
            {'k': 3, 'logits': torch.tensor([0.7, 0.1, 0.2]), 'expected': torch.tensor([0.7, 0.1, 0.2])},
        )
        for case in test_cases:
            config = SamplerConfig(temperature=1.0, k=case["k"], p=0, repetition_penalty=1.0)
            sampler = Sampler(config, device=torch.device("cpu"))
            filtered_logits = sampler.apply_top_k_filter(case["logits"])
            np.testing.assert_array_equal(case["expected"].numpy(), filtered_logits.numpy())


class SamplerSingleStackTest(unittest.TestCase):
    class ModelStub(object):
        def __init__(self, batch_size=1, vocabulary_size=5):
            self.batch_size = batch_size
            self.vocabulary_size = vocabulary_size

        def decode(self, _):
            return self(_)

        def __call__(self, _):
            return (0.5 * torch.ones((self.batch_size, 2, self.vocabulary_size)),)

    def test_wrong_k_value(self):
        model = self.ModelStub(vocabulary_size=5)
        sampler = generate.new_sampler(model, k=10)
        with self.assertRaises(RuntimeError):
            _ = sampler.generate_sequence(length=1)

    def test_forward_pass_and_output_length(self):
        models = {
            "XLNet": XLNetLMHeadModel(XLNetConfig()),
            "generic": self.ModelStub(),
        }
        models_kwargs = {
            "XLNet": {} ,
            "generic": {},
        }
        prompt = [1, 2, 3]
        generated_length = 5
        expected_length_with_prompt = 8
        expected_length_without_prompt = 5

        for name, model in models.items():
            kwargs = models_kwargs[name]
            sampler = generate.new_sampler(model, k=2, p=0.5, repetition_penalty=2)
            output1 = sampler.generate_sequence(length=generated_length, **kwargs)
            output2 = sampler.generate_sequence(prompt_ids=prompt, length=generated_length, **kwargs)
            self.assertEqual(len(output1), expected_length_without_prompt)
            self.assertEqual(len(output2), expected_length_with_prompt)


if __name__ == "__main__":
    unittest.main()
