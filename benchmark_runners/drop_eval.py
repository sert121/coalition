from deepeval.benchmarks import DROP
from deepeval.benchmarks.tasks import DROPTask



from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import vllm
from huggingface_hub import snapshot_download
from vllm import SamplingParams


class CustomLM(DeepEvalBaseLLM):

    def __init__(self,
                 model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                 lora_path=None):

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

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"model name {self.model_name}"


def run_drop_benchmark(config):

    # Define benchmark with specific tasks and number of code generations

    # provide custom config for the class
    benchmark = DROP()
    model = CustomLM(model_name = 'meta-llama/Meta-Llama-3-8B-Instruct' )

    benchmark.evaluate(model = model) # no parameter k  

    print(benchmark.overall_score)

    print("saving benchmark predictions to csv")
    df = benchmark.predictions
    df.to_csv('drop_benchmark_predictions.csv')
