from deepeval.benchmarks import GSM8K
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import vllm

from vllm import SamplingParams

# Define benchmark with specific tasks and number of code generations
# take user input
# adapt the benchmark outputs !TODO


class GSM_Custom_LM(DeepEvalBaseLLM):

    def __init__(
            self,
            config=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lora_path=None):
        # just take the quantized model path (assume the model is already quantized)

        self.config = config
        self.model_name = model_name

        self.sampling_params = SamplingParams(
        **(vars(config.sampling_params)), logprobs=1)
            

        if lora_path is not None:
            try:
                adapter_lora_path = snapshot_download(
                    repo_id=self.config.lora_path)
                llm = vllm.LLM(model=self.config.model_name, enable_lora=True)
                self.lora_enabled = True
                self.lora_request = LoRARequest("lora_adapter", 1, lora)
            except Exception as e:
                print(e)
                print("Lora adapter not found, using default model")
                llm = vllm.LLM(model=self.config.model_name)
                self.lora_enabled = False

        else:
            llm = vllm.LLM(model=self.config.model_name)
            self.lora_enabled = False

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.model = llm
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # generate a single prompt output
        model = self.load_model()
        if 'llama' in self.config.model_name:
            tokenizer = self.model.get_tokenizer()

            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': f'conversations'}],
                tokenize=False,
            )
            self.sampling_params = SamplingParams(
                    ** self.config.sampling_params,
                    logprobs=1,
                    stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
                ) # override the sampling params to include the stop token


        # generate the model output
        if self.lora_enabled:
            output = model.generate(prompt, self.sampling_params,
                                    self.lora_request)
        else:
            print("-- prompt input:-- \n", prompt)
            output = model.generate(prompt, self.sampling_params)

            print(" -- prompt output: --\n", output[0].outputs[0].text)
            print(" === end of prompt output === ")

        generated_text = output[0].outputs[0].text
        # return the output
        return generated_text

    async def a_generate(self, prompt: str) -> str:  #!TODO: fix
        return self.generate(prompt)

    def get_model_name(self):
        return f"{self.model_name}"


def run_gsm_benchmark(config):

    benchmark = GSM8K(
        n_shots=config.n_shots,
        enable_cot=config.enable_cot,
        n_problems=config.n_problems  # number of problems
    )

    benchmark.evaluate(model=GSM_Custom_LM(config))

    print(benchmark.overall_score)
    return benchmark.overall_score, benchmark.predictions
