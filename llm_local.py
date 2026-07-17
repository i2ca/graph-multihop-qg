from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llm_api import LlmApi

class LlmLocalApi(LlmApi):

    def __init__(self):
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        #model_name = "./models/finetuned_llama_1b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
        self.model = torch.compile(self.model)
        self.model.eval()

    def query(self, prompt: str):
        input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = input.input_ids.shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **input,
                max_new_tokens=512,
                temperature=None,
                top_p=None,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = output[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text