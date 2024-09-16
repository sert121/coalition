from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

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
                 config=None,
                 model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                 lora_path=None):
        # just take the quantized model path (assume the model is already quantized)

        self.config = config
        self.model_name = model_name

        self.sampling_params = SamplingParams(**(vars(config.sampling_params)),
                                              logprobs=1)

        if lora_path is not None:
            try:
                adapter_lora_path = snapshot_download(
                    repo_id=self.config.lora_path)
                llm = vllm.LLM(model=self.config.model_name, tokenizer=self.config.model_name, enable_lora=True)


                self.lora_enabled = True
                self.lora_request = LoRARequest("lora_adapter", 1, lora)
            except Exception as e:
                print(e)
                print("Lora adapter not found, using default model")
                llm = vllm.LLM(model=self.config.model_name)
                self.lora_enabled = False

        else:
            llm = vllm.LLM(model=self.config.model_name, tokenizer=self.config.model_name,)

            self.lora_enabled = False

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

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

    def batch_generate(self, prompts: list) -> list:

        model = self.load_model()
        # generate the model output
        if self.lora_enabled:
            outputs = model.generate(prompts, self.sampling_params,
                                    self.lora_request)
        else:
            print("-- prompt input:-- \n", prompt)
            outputs = model.generate(prompt, self.sampling_params)

            print(" -- prompt sample output: --\n", output[0].outputs[0].text)
            print(" === end of sample prompt output === ")

        # extract the model outputs
        generated_texts = [output.outputs[0].text for output in outputs]
        # return the output
        return generated_texts

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"{self.model_name}"


def run_mmlu_benchmark(config):

    if config.tasks is not None:
        benchmark = MMLU(n_shots=config.n_shots, tasks=tasks)
    else:
        benchmark = MMLU(n_shots=config.n_shots)
    
    model = CustomLM(model_name=config.model_name)

    if batch_size is not None:
        benchmark.evaluate(model=model,
                           batch_size=config.batch_size,
                           k=config.k)
    else:
        benchmark.evaluate(model=model)  # no parameter k

    print(benchmark.overall_score)
    print("saving benchmark predictions to csv")

    df = benchmark.predictions
    df.to_csv('mmlu_benchmark_predictions.csv')

    return benchmark.overall_score, benchmark.predictions
