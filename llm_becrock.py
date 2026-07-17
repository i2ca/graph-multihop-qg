import json
import boto3
from botocore.exceptions import ClientError
from llm_api import LlmApi

class LlmBedrockLlamaApi(LlmApi):

    def __init__(self):
        self.bedrock_runtime_client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

    def query(self, prompt):
        try:
            body = {
                "prompt": prompt,
                "temperature": 0,
                "top_p": 1.0,
                "max_gen_len": 32,
            }
            response = self.bedrock_runtime_client.invoke_model(
                modelId="us.meta.llama3-2-1b-instruct-v1:0", body=json.dumps(body)
            )
            response_body = json.loads(response["body"].read())
            response_text = response_body['generation']
            return response_text
        except ClientError:
            print("Couldn't invoke Llama 2 -----------------------------------------------------------------")
            raise



class LlmBedrockMixtralApi(LlmApi):

    def __init__(self):
        self.bedrock_runtime_client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

    def query(self, prompt):
        try:
            body = {
                "prompt": prompt,
                "temperature": 0.1,
                "top_p": 0.9
            }
            response = self.bedrock_runtime_client.invoke_model(
                modelId="mistral.mistral-7b-instruct-v0:2", body=json.dumps(body)
            )
            response_body = json.loads(response["body"].read())
            response_text = response_body['outputs'][0]['text']
            return response_text

        except ClientError:
            print("Couldn't invoke Mixtral")
            raise



class LlmBedrockCommandApi(LlmApi):

    def __init__(self):
        self.bedrock_runtime_client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

    def query(self, prompt):
        try:
            body = {
                "prompt": prompt,
                "temperature": 0.1
            }
            response = self.bedrock_runtime_client.invoke_model(
                modelId="cohere.command-text-v14", body=json.dumps(body)
            )
            response_body = json.loads(response["body"].read())
            response_text = response_body['generations'][0]['text']
            return response_text

        except ClientError:
            print("Couldn't invoke Mixtral")
            raise



class LlmBedrockTitanApi(LlmApi):

    def query(self, prompt):
        try:
            body = {
                "inputText": prompt
            }
            response = self.bedrock_runtime_client.invoke_model(
                modelId="amazon.titan-text-express-v1", body=json.dumps(body)
            )
            response_body = json.loads(response["body"].read())
            response_text = response_body['results'][0]['outputText']
            return response_text
        except ClientError:
            print("Couldn't invoke Titan -----------------------------------------------------------------")
            raise

    def call_api(self, prompt):
        return self.call_titan_api(prompt)