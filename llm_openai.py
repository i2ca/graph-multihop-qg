from openai import OpenAI
from llm_api import LlmApi

class LlmOpenaiApi(LlmApi):

    def __init__(self, model="gpt-4o-mini", api_key_file="api_key.txt"):
        self.model = model
        self.api_key = self.get_api_key(api_key_file)
        self.client = OpenAI(api_key=self.api_key)

    def get_api_key(self, file_path):
        with open(file_path, "r") as file:
            api_key = file.readline().rstrip('\n')
        return api_key

    def query(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            n=1,
            temperature=0.0,
            top_p=1.0
        )
        completion = response.choices[0].message.content.strip()
        return completion