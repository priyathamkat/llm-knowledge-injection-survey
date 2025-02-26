from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pydantic import BaseModel


class SFTSample(BaseModel):
    prompt: str
    completion: str


class SimpleWordTask(ABC):
    def __init__(self, seed: int = 128):
        self._rng = np.random.default_rng(seed=seed)
        self._words = None

    @property
    def words(self) -> list[str]:
        """List of words to sample from. This is lazily loaded from a file."""
        if self._words is None:
            words_path = Path(__file__).parent / "words.txt"
            with open(words_path, "r") as f:
                self._words = f.read().splitlines()
        return self._words

    @abstractmethod
    def sft_sample(self, n: int) -> SFTSample:
        """SFT samples for a simple word task."""
        raise NotImplementedError


class WordLengthTask(SimpleWordTask):
    def sft_sample(self, num_samples: int) -> list[SFTSample]:
        """SFT samples for word length task."""
        prompt_template = "What is the length of the word: {word}? Output only the number and nothing else."
        completion_template = "{length}"
        words = self._rng.choice(self.words, size=num_samples)
        samples = []
        for word in words:
            prompt = prompt_template.format(word=word)
            completion = completion_template.format(length=len(word))
            samples.append(SFTSample(prompt=prompt, completion=completion))
        return samples


class NthLetterTask(SimpleWordTask):
    def sft_sample(self, num_samples: int) -> list[SFTSample]:
        """SFT samples nth letter task."""
        prompt_template = "What is the {n}{n_suffix} letter of the word: {word}? Output only the letter and nothing else."
        completion_template = "{letter}"
        words = self._rng.choice(self.words, size=num_samples)
        samples = []
        for word in words:
            n = self._rng.integers(1, len(word) + 1)
            match n:
                case 1:
                    n_suffix = "st"
                case 2:
                    n_suffix = "nd"
                case 3:
                    n_suffix = "rd"
                case _:
                    n_suffix = "th"
            prompt = prompt_template.format(n=n, n_suffix=n_suffix, word=word)
            completion = completion_template.format(letter=word[n - 1])
            samples.append(SFTSample(prompt=prompt, completion=completion))
        return samples
