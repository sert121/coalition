from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

from deepeval.models import DeepEvalBaseLLM
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import vllm

# Define benchmark with specific tasks and number of code generations
benchmark = MMLU(
    n_shots =1 # one can change n_shots !TODO: load n_shot from user
)

class CustomLM(DeepEvalBaseLLM):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # load quantization config from json if any !TODO

        # load the sampling params
        self.sampling_params = SamplingParams(temperature=0)

        # load lora config from json if any !TODO

        # insert vllm code to load model (with above c)
        llm = vllm.LLM(model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = llm
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # generate a single prompt output
        model = self.load_model()

        output = model.generate(prompt, self.sampling_params)
        # generate the model outputs using vllm
        generated_text = output.outputs[0].text
        # return the output
        return generated_text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "model name "

benchmark.evaluate(model = CustomLM()) # no parameter k 
# benchmark.evaluate(model=gpt_4, k=1)
print(benchmark.overall_score)
