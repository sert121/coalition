from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

import vllm, re
from huggingface_hub import snapshot_download
from vllm import SamplingParams


class CustomLM(DeepEvalBaseLLM):

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", lora_path=None):
        # load quantization config from json if any !TODO
        # just take the quantized model path (assume the model is already quantized)

        self.model_name = model_name
        # load the sampling params
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1024, skip_special_tokens=True)

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


    def extract_first_function(self,text):
        # Regular expression to match a Python function definition
        pattern = r'def\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+\s*)?:\n(?:(?:\s+.*\n)+)'

        # Find the first match in the text
        match = re.search(pattern, text)

        if match:
            # Return the matched function definition
            return match.group(0).rstrip()
        else:
            return "No function definition found."


    def load_model(self):
        return self.model

    def filter_code(self,completion: str) -> str:
        # The program tends to overwrite, we only take the first function
        completion = completion.lstrip("\n")
        return completion.split("\n\n")[0]


    def fix_indents(self,text: str) -> str:
        return text.replace("\t", "    ")

    def generate(self, prompt: str) -> str:
        # generate a single prompt output
        model = self.load_model()

        # generate the model output
        if self.lora_enabled:
            output = model.generate(prompt, self.sampling_params,
                                    self.lora_request)
        else:
            print("-- prompt input:--\n",prompt)
            output = model.generate(prompt, self.sampling_params)

            print(" -- start of prompt output: --\n", output[0].outputs[0].text)
            print("\n === end of prompt output === \n")
        # extract the model outputs
        generated_text = output[0].outputs[0].text
        # return the output
        return generated_text

    def generate_samples(self, prompt: str, n: int, temperature: float):
        model = self.load_model()
        prompts = [prompt for _ in range(n)]
        print("-- prompt input:-- \n", prompts[0])

        if self.lora_enabled:
            generated_outputs = model.generate(prompt,
                                     self.sampling_params,
                                    self.lora_request)
        else:
            generated_outputs = model.generate(prompts, self.sampling_params)

        # extract text
        generated_outputs = [output.outputs[0].text for output in generated_outputs]

        # fix generations
        generated_outputs = [self.filter_code(output) for output in generated_outputs]

        generated_outputs = [output.replace("    ", "\t") for output in generated_outputs]

        # generated_outputs = [self.fix_indents(output) for output in generated_outputs]

        # extract the first def
        generated_outputs = [self.extract_first_function(output) for output in generated_outputs]

        responses_list = []
        for i in range(n):
            generated_text = generated_outputs[i]
            print(" -- start of prompt output: --\n", generated_text)
            print("\n === end of prompt output === \n")
            responses_list.append(generated_text)

        return responses_list

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"{self.model_name}"


def run_humaneval_benchmark(config):
    # parse and load the config !TODO

    # Define benchmark with specific tasks and number of code generations
    benchmark = HumanEval(
        n=1 # take n from user
    )

    model = CustomLM(model_name='meta-llama/Meta-Llama-3-8B-Instruct')
    benchmark.evaluate(model=model, k=1)
    # benchmark.evaluate(model=gpt_4, k=1)
    print(benchmark.overall_score)
    # ideally return the score, and the log probs, and also maybe the saved evals
    return benchmark.overall_score, benchmark.predictions
