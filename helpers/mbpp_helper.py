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

from typing import Dict, Iterable

from tqdm import tqdm
import os
import torch
import json

import vllm
from huggingface_hub import snapshot_download
from vllm import SamplingParams



def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


def create_mbpp_instruction(example):
    description = example["text"]
    test_cases = example["test_list"]
    prompt = "You are an expert Python programmer, and you need to respond with code to complete is your task: {description} Your code should pass these tests:\n\n{tests}\n."

    # prompt = "You are an expert Python programmer, and you need to respond with code to complete is your task. Your code should pass certain tests. You shall also be provided with some steps to help you solve the problem.  \n Task: {description}. \n Tests: \n\n{tests}\n."
    # prompt = "Task:\n{description}\nTests:\n{tests}\n." # comment if not using gpt

    #prompt modified for mbpp codellama
    # prompt =  " Remember to respond with only the function definition, nothing else. \nTask: {description} \nTests:\n {tests}\nCode:"
    # prompt =  "You are an expert Python programmer, and you need to respond with code to complete is your task. "

    #promptfor mbpp codellama (zero-shot steps)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided steps to generate more accurate code. \nTask: {description} \nTests:\n {tests}"

    # prompt for mbpp codellama (one shot steps)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided steps to generate more accurate code. You are provided an example to understand the structure. \nTask: {description} \nTests:\n {tests}"

    #prompt for mbpp codellama (zero shot pseudo code)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided pseudocode to generate more accurate code. \nTask: {description} \nTests:\n {tests}"

    # prompt for mbpp codellama (one shot pseudo code)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided pseudocode to generate more accurate code. You are provided an example to understand the structure. \nTask: {description} \nTests:\n {tests}"

    # prompt = : "You are an expert Python programmer, and you need to respond with code to complete the following task. Remember to respond with only the function definition, nothing else. You shall be provided pseudocode to generate more accurate code. You are provided an example to understand the structure. \nTask: {description} \nTests:\n {tests}"

    instruction = prompt.format(description=description,
                                tests="\n".join(test_cases))
    return instruction

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
    lora_enabled: bool = False,
    max_tokens: int = 100,
):
    # TODO: cleanup this function and merge this with the caller functions that calls run eval mbpp oOR maybe get rid of the generate batch function completely and switch it out with vllm code

    task_ids, instructions = load_data_mbpp()
    # problems = [prompter.generate_prompt(instruction) for instruction in instructions] # uncoment if not using gpt
    problems = instructions  # comment if not using gpt
    problems = problems[:length]
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    #regulate this using config
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    # TODOL: add lora request
    counter = 0
    for i, prompt in enumerate(problems):
        # if format_tabs:
        #     prompt = problems[task_ids[i]]["prompt"].replace("    ", "\t")
        # else:
        #     prompt = problems[task_ids[i]]["prompt"]

        batch_completions = generate_batch_completion(model=model, tokenizer=tokenizer, prompt=prompt,
                                                      sampling_params = sampling_params,
                                                    num_samples_per_task=num_samples_per_task,
                                                        lora_enabled=False, counter=counter)
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


def generate_batch_completion(model, tokenizer,prompt,
                              num_samples_per_task,
                             sampling_params, lora_enabled,
                             counter, lora_request = None, filter_use = True) -> list[str]:
    # to account for mbpp -delay , mbpp starts testing on samples from 10th sample onwards
    counter = counter + 10
    
    # change this completely. switch to vllm
    input_batch = [prompt for _ in range(num_samples_per_task)]

    # generate the model output
    if lora_enabled:
        batch_completions = model.generate(input_batch, sampling_params,
                                           lora_request)
    else:
        print("-- prompt input:-- \n", input_batch[0])
        batch_completions = model.generate(prompt, sampling_params)

        print(" -- prompt output: --\n", batch_completions[0].outputs[0].text)
        print(" === end of prompt output === ")

    # extract the model outputs
    generated_text = batch_completions[0].outputs[0].text
    batch_completions = [
        output.outputs[0].text for output in batch_completions
    ]
    if filter_use:
        batch_completions = [
            filter_code(completion) for completion in batch_completions
        ]

    return batch_completions


if __name__ == "__main__":
    # adjust for n = 10 etc
    parser = argparse.ArgumentParser(description="Eval model")
    # parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-34b-Instruct-hf') # uncomment this line for 34b model

    parser.add_argument('--model_name',
                        type=str,
                        default='codellama/CodeLlama-7b-Instruct-hf'
                        )  # uncomment this line for 7b model
    parser.add_argument('--length', type=int, default=100)
    args = parser.parse_args()
    print(args)

    num_samples_per_task = 5

    # output path
    out_path = "results/" + args.model_name.split(
        '/')[0] + "/mbpp_" + args.model_name.split('/')[1] + '_' + str(
            args.length) + ".jsonl"
    print("Out path: ", out_path)
    os.makedirs("results/" + args.model_name.split('/')[0], exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading model...")
    # switch to vllm model
    model = vllm.LLM(args.model_name)

    run_eval_mbpp(
        model=model,
        tokenizer=tokenizer,
        num_samples_per_task=num_samples_per_task,
        out_path=out_path,
        generate_batch_completion=generate_batch_completion,
        length=args.length,  #number of problems
        format_tabs=True,
        lora_enabled=False)
