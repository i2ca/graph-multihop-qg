from __future__ import annotations

import os

from llm_api import LlmApi
from google import genai

class LlmGeminiApi(LlmApi):

    def __init__(self, model: str = "gemini-2.5-flash", api_key_file="api_key_gemini.txt") -> None:
        self.model = model
        self.api_key_gemini = self.get_api_key(api_key_file)
        self.client = genai.Client(api_key=api_key_gemini)
        
    def get_api_key(self, file_path):
        with open(file_path, "r") as file:
            api_key = file.readline().rstrip('\n')
        return api_key

    def query(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response")
