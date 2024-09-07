import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import os
import torch
import json

import vllm

def load_data_mbpp():
    dataset_name = 'mbpp'
    data_path_mapping = {
            "mbpp": "./data/mbpp.jsonl",
            "humaneval": "./data/HumanEval.jsonl.gz"
            }
    data_path = data_path_mapping[dataset_name]
    # data_path = "/home/mila/m/megh.thakkar/CodeCapybara/main/data/mbpp.jsonl"
    data = []
    if data_path.endswith(".jsonl.gz"):
        with gzip.open(data_path, "rt") as f:
            data = [json.loads(line) for line in f]
    elif data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
    else:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

    instructions = []
    task_ids = []
    if dataset_name == "mbpp":
        data = list(filter(lambda x: x["task_id"] in range(11, 511), data))
        instructions = list(map(create_mbpp_instruction, data))
        task_ids = list(map(lambda x: x["task_id"], data))
    else:
        task_ids = [ex["task_id"] for ex in data]
        instructions = [ex["prompt"] for ex in data]

    return task_ids, instructions



# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def run_eval_mbpp(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion,
    length: int, #number of problems
    format_tabs: bool = False,
):
    # TODO: cleanup this function and merge this with the caller functions that calls run eval mbpp oOR maybe get rid of the generate batch function completely and switch it out with vllm code

    task_ids, instructions = load_data_mbpp()
    # problems = [prompter.generate_prompt(instruction) for instruction in instructions] # uncoment if not using gpt
    problems = instructions  # comment if not using gpt
    problems = problems[:length]
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    counter = 0
    for i, prompt in enumerate(problems):
        # if format_tabs:
        #     prompt = problems[task_ids[i]]["prompt"].replace("    ", "\t")
        # else:
        #     prompt = problems[task_ids[i]]["prompt"]

        batch_completions = generate_batch_completion(model, tokenizer, prompt,
                                                      num_samples_per_task,
                                                      counter)
        # print (batch_completions)
        for sample in batch_completions:
            result = dict(
                task_id=task_ids[i],
                # trg_prediction=prompter.get_response(sample),
                trg_prediction=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)
        counter += 1

    write_jsonl(out_path, samples)



def load_json(file_path="./mbpp_examples_magicoder_reform_v1.json"):
    with open(file_path, "r") as json_file:
        data_dict = json.load(json_file)

    list_prompts = []
    for k, v in data_dict.items():
        list_prompts.append(v)
    # sorted_dict = {int(key): value for key, value in sorted(data_dict.items())}
    # sorted_value_list = [value for key, value in sorted_dict.items()]
    return list_prompts


# ZERO SHOT + STEPS
def construct_codellama_prompt(problem, thought):
    problem = problem.split('[/INST]')[0]
    PROMPT = f'{problem} \nSteps:\n {thought} \nCode:[/INST]'
    return PROMPT


# ZERO SHOT + PSEUDOCODE
def construct_codellama_pseudo_prompt(problem, thought):
    problem = problem.split('[/INST]')[0]
    PROMPT = f'{problem} \nPseudocode:\n {thought} \nCode:\n[/INST]'
    return PROMPT



# ONE SHOT + PSSEUDOCODE
def construct_codellama_pseudo_prompt_example(problem, thought):
    problem = problem.split('[/INST]')[0]
    example = '''
    EXAMPLE STARTS HERE
        Task: Write a program that extracts all substrings of length n from a given string.
        Tests:
        assert find_substrings("abc", 2) == ["ab", "bc"]
        assert find_substrings("abc", 3) == ["abc"]
        assert find_substrings("abc", 4) == []
        
        Pseudocode:
        function extract_substrings(string, n)
            # Initialize an empty list for substrings
            Initialize substrings as an empty list

            # Loop from start to the point where substring of length n can be extracted
            for every index in the string
                # Add substring of length n to the list
                append string[i:i + n] to substrings

            # Return the list of substrings
            return the substrings
        Code:
        def find_substrings(string, n):
            substrings = []
            for i in range(len(string) - n + 1):
                substrings.append(string[i:i + n])
            return substrings
    EXAMPLE ENDS HERE
    '''

    # used for benchmarking
    PROMPT = example + f'{problem} \nPseudocode:\n{thought} \nCode:\n[/INST]'

    # v2 used for experimentation
    PROMPT = f'{problem}\n {example} \nPseudocode:\n{thought} \nCode:\n[/INST]'
    return PROMPT



# ONE SHOT  + STEPS PROMPT
def construct_codellama_prompt_steps(problem, thought):
    problem = problem.split('[/INST]')[0]
    example = '''
EXAMPLE STARTS HERE
    Task: Write a program that extracts all substrings of length n from a given string.
    Tests:
    assert find_substrings("abc", 2) == ["ab", "bc"]
    assert find_substrings("abc", 3) == ["abc"]
    assert find_substrings("abc", 4) == []
    
    Steps:
    1. Initialize an empty list for substrings
    2. Loop from start to the point where substring of length n can be extracted
        a. Add substring of length n to the list
    3. Return the final list of substrings that was created
    Code:
    def find_substrings(string, n):
        substrings = []
        for i in range(len(string) - n + 1):
            substrings.append(string[i:i + n])
        return substrings
EXAMPLE ENDS HERE     
'''
    PROMPT = example + \
        f'{problem} \nSteps:\n {thought} \nCode:\n[/INST]'  # v1 that is used in baselines

    # v2
    PROMPT = f'{problem}\n {example} \nSteps:\n {thought} \nCode:\n[/INST]'

    return PROMPT


def generate_batch_completion(model: vllm.LLM, tokenizer: PreTrainedTokenizer,
                              prompt, sampling_params, lora_enabled,
                              lora_request, batch_size, counter) -> list[str]:
    # to account for mbpp -delay , mbpp starts testing on samples from 10th sample onwards
    counter = counter + 10

    # change this completely. switch to vllm
    input_batch = [prompt for _ in range(batch_size)]

    # generate the model output
    if self.lora_enabled:
        batch_completions = model.generate(input_batch, self.sampling_params,
                                           self.lora_request)
    else:
        print("-- prompt input:-- \n", input_batch[0])
        batch_completions = model.generate(prompt, self.sampling_params)

        print(" -- prompt output: --\n", batch_completions[0].outputs[0].text)
        print(" === end of prompt output === ")
    # extract the model outputs
    generated_text = batch_completions[0].outputs[0].text

    # batch_completions = tokenizer.batch_decode([ids for ids in generated_ids],
    #                                            skip_special_tokens=True,
    #                                            ignore_tokenization_space=True)

    batch_completions = [
        output.outputs[0].text for output in batch_completions
    ]

    return batch_completions


if __name__ == "__main__":
    # adjust for n = 10 etc
    parser = argparse.ArgumentParser(description="Eval model")
    # parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-34b-Instruct-hf') # uncomment this line for 34b model

    parser.add_argument('--model_name', type=str,
                        default='codellama/CodeLlama-7b-Instruct-hf')    # uncomment this line for 7b model
    parser.add_argument('--length', type=int, default=100)
    args = parser.parse_args()
    print(args)

    num_samples_per_task = 5

    # output path
    out_path = "results/" + args.model_name.split('/')[0] + "/mbpp_" + args.model_name.split('/')[
        1] + '_' + str(args.length) + ".jsonl"
    print("Out path: ", out_path)
    os.makedirs("results/" + args.model_name.split('/')[0], exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )

    print("Loading model...")
    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )

    run_eval_mbpp(model, tokenizer, num_samples_per_task, out_path,
                  generate_batch_completion, args.length, True)
