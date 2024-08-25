from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask

from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import vllm
from huggingface_hub import snapshot_download


# Define benchmark with specific tasks and number of code generations
benchmark = TruthfulQA()

class CustomLM(DeepEvalBaseLLM):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", lora_path=None):

        # load quantization config from json if any !TODO
        # just take the quantized model path (assume the model is already quantized)
        self.model_name = model_name
        # load the sampling params
        self.sampling_params = SamplingParams(temperature=0)

        # load lora config from json if any !TODO
        # if the lora adapters are present, take the path
        if lora_path is not None:
            adapter_lora_path = snapshot_download(repo_id=sql_lora_path)
            llm = vllm.LLM(model=model_name, enable_lora=True)
            self.lora_enabled = True
            self.lora_request = LoRARequest("lora_adapter", 1, lora)

        else:
            # insert vllm code to load model (with above c)
            llm = vllm.LLM(model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = llm
        self.tokenizer = tokenizer


    def generate(self, prompt: str) -> str:
        # generate a single prompt output
        model = self.model

        # generate the model output
        if self.lora_enabled:
            output = model.generate(prompt, 
            self.sampling_params, 
            self.lora_request)
        else:
            output = model.generate(prompt, self.sampling_params)

        # extract the model outputs
        generated_text = output.outputs[0].text
        # return the output
        return generated_text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"model name {self.model_name}"

benchmark.evaluate(model = CustomLM()) # no parameter k 
# benchmark.evaluate(model=gpt_4, k=1)
print(benchmark.overall_score)
