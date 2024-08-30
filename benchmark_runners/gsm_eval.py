from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import vllm

# Define benchmark with specific tasks and number of code generations
# take user input
# adapt the benchmark outputs !TODO


class CustomLM(DeepEvalBaseLLM):

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        # load quantization config from json if any !TODO
        # just take the quantized model path (assume the model is already quantized)

        self.model_name = model_name
        # load the sampling params
        self.sampling_params = SamplingParams(temperature=0, max_tokens=50)

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

    async def a_generate(self, prompt: str) -> str:  #!TODO: fix
        return self.generate(prompt)

    def get_model_name(self):
        return f"{self.model_name}"

def run_gsm_benchmark(config):

    # parse the config please ! TODO
    benchmark = GSM8K(
        n_shots = 0,
        enable_cot = 1,
        n_problems = 1 # number of problems
    )

    benchmark.evaluate(model=CustomLM())
    print(benchmark.overall_score)
