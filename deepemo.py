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


class CustomLlama3_8B(DeepEvalBaseLLM):

    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            # quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct")

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=100,
            do_sample=True,
            # top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)

    def generate_samples(self, prompt: str, n: int, temperature: float):
        chat_model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=chat_model,
            tokenizer=self.tokenizer,
            temperature=1,
            use_cache=True,
            device_map="auto",
            # max_length=100,
            max_new_tokens = 100,
            do_sample=True,
            # top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        responses_list = []
        for i in range(n):
            response = pipeline(prompt)
            print("\n--- the response is : ---\n")
            print(response)
            
            response = response[0]["generated_text"]
            responses_list.append(response)

        # print("the response list is :")
        # print(responses_list)
        return responses_list

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"


benchmark.evaluate(model = CustomLlama3_8B(), k=1)
# benchmark.evaluate(model=gpt_4, k=1)
print(benchmark.overall_score)
