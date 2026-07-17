from abc import ABC, abstractmethod

class LlmApi(ABC):

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass