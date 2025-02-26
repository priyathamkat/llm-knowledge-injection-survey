from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pydantic import BaseModel


class Sample(BaseModel):
    prompt: str
    completion: str


class SimpleWordTask(ABC):
    def __init__(self, seed: int = 128):
        self._rng = np.random.default_rng(seed=seed)
        self._words = None

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Template prompt for the task."""
        raise NotImplementedError

    @property
    @abstractmethod
    def completion_template(self) -> str:
        """Template completion for the task."""
        raise NotImplementedError

    @property
    def words(self) -> list[str]:
        """List of words to sample from. This is lazily loaded from a file."""
        if self._words is None:
            words_path = Path(__file__).parent / "words.txt"
            with open(words_path, "r") as f:
                self._words = f.read().splitlines()
        return self._words

    @abstractmethod
    def sample(self, n: int) -> Sample:
        """Sample a simple word task."""
        raise NotImplementedError


class WordLengthTask(SimpleWordTask):
    @property
    def prompt_template(self):
        return "What is the length of the word: {word}? Output only the number and nothing else."

    @property
    def completion_template(self):
        return "{length}"

    def sample(self, num_samples: int) -> list[Sample]:
        """Sample word length tasks."""
        words = self._rng.choice(self.words, size=num_samples)
        samples = []
        for word in words:
            prompt = self.prompt_template.format(word=word)
            completion = self.completion_template.format(length=len(word))
            samples.append(Sample(prompt=prompt, completion=completion))
        return samples


class NthLetterTask(SimpleWordTask):
    @property
    def prompt_template(self):
        return "What is the {n}{n_suffix} letter of the word: {word}? Output only the letter and nothing else."

    @property
    def completion_template(self):
        return "{letter}"

    def sample(self, num_samples: int) -> list[Sample]:
        """Sample nth letter tasks."""
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
            prompt = self.prompt_template.format(n=n, n_suffix=n_suffix, word=word)
            completion = self.completion_template.format(letter=word[n - 1])
            samples.append(Sample(prompt=prompt, completion=completion))
        return samples
