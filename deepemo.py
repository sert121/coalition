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

        # model_4bit = AutoModelForCausalLM.from_pretrained(
        #     "meta-llama/Meta-Llama-3-8B-Instruct",
        #     # quantization_config=quantization_config,
        # )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct")

        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        model = vllm.LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )

        self.sampling_params = SamplingParams(temperature=0, max_tokens=100)
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def load_vllm_model(self):
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

    def generate_samples_old(self, prompt: str, n: int, temperature: float):
        chat_model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=chat_model,
            tokenizer=self.tokenizer,
            temperature=1,
            use_cache=True,
            device_map="auto",
            # max_length=100,
            max_new_tokens=100,
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

    def generate_samples(self, prompt: str, n: int, temperature: float):

        prompts = [prompt for i in range(n)]
        outputs = self.model.generate(prompts=prompts,
                                      sampling_params=self.sampling_params,
                                      prompt_token_ids=prompt_token_ids)

        responses_list = []
        for i in range(n):
            output = outputs[i].outputs[0].text
            responses_list.append(output)

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
