from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

# Define benchmark with specific tasks and number of code generations
benchmark = HumanEval(
    tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS, HumanEvalTask.SORT_NUMBERS],
    n=1
)

class CustomLM(DeepEvalBaseLLM):

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        # load quantization config from json if any !TODO
        # just take the quantized model path (assume the model is already quantized)
        
        self.model_name = model_name
        # load the sampling params
        self.sampling_params = SamplingParams(temperature=0, max_tokens=15)

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
            self.lora_enabled = False
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = llm
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model


    def generate(self, prompt: str) -> str:
        # generate a single prompt output
        model = self.load_model()

        # generate the model output
        if self.lora_enabled:
            output = model.generate(prompt, self.sampling_params,
                                    self.lora_request)
        else:
            print("-- prompt input:-- \n", prompt)
            output = model.generate(prompt, self.sampling_params)

            print(" -- prompt output: --\n", output[0].outputs[0].text)
            print(" === end of prompt output === ")
        # extract the model outputs
        generated_text = output[0].outputs[0].text
        # return the output
        return generated_text

    def generate_samples(self, prompt: str, n: int, temperature: float):
        model = self.load_model()

        if self.lora_enabled:
            outputs = model.generate(prompt,
                                     self.sampling_params,
                                    self.lora_request)
        else:
            generated_outputs = model.generate(prompt, self.sampling_params)

        responses_list = []
        for i in range(n):
            generated_text = generated_outputs[i].outputs[0].text
            responses_list.append(generated_text)

        return responses_list

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"

model = CustomLM(model_name = 'meta-llama/Meta-Llama-3-8B-Instruct')
benchmark.evaluate(model =model, k=1)
# benchmark.evaluate(model=gpt_4, k=1)
print(benchmark.overall_score)
