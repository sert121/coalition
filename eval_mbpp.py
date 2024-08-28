
import os
import fire
import json
import regex
import signal
import tempfile
import threading
import subprocess
import collections
from glob import glob
from tqdm import tqdm
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Union, List
import numpy as np
import argparse 
import re

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode

class MBPPGoogleDataset(object):
    def __init__(self, path="dataset/mbpp/mbpp.jsonl", mode="function_name"):
        raw_data = sorted(
            [json.loads(x) for x in open(path)], key=lambda x: x["task_id"]
        )
        for i, data_item in enumerate(raw_data):
            assert data_item["task_id"] == i + 1
        self.raw_data = collections.defaultdict()
        self.mode = mode
        # 374 for training, 100 heldout, 500 test
        self.raw_data["train"] = raw_data[:10] + raw_data[510:]
        self.raw_data["test"] = raw_data[10:510]
        # data for codex collector, in input-output-info format
        self.data = collections.defaultdict()
        for split in self.raw_data:
            self.data[split] = self.extract_data(self.raw_data[split], mode)

    @staticmethod
    def extract_data(raw_data, mode):
        if mode == "function_name":
            get_function_name = lambda test_example: regex.match(
                "assert [\(]*([^\(]+)\(", test_example
            ).group(1)
            info = [get_function_name(x["test_list"][0]) for x in raw_data]
        elif mode == "assertion":
            info = [x["test_list"][0] for x in raw_data]
        elif mode == "assertion-full":
            info = [x["test_list"] for x in raw_data]
        else:
            raise Exception(f"Mode {mode} not supported.")
        nls = [x["text"] for x in raw_data]
        codes = [x["code"] for x in raw_data]
        return list(zip(nls, codes, info))

def evaluate_one_mbpp(args, tempdir, references, timeout):
    i, item = args
    task_id = item["task_id"]
    print (task_id)
    ref = references[task_id - 10 - 1]
    test_cases = ref["test_list"]
    test_setups = ref["test_setup_code"]
    code = item["trg_prediction"]
    # write code to file
    with open(f"{tempdir.name}/code-{i}.py", "w") as fout:
        print(code, file=fout)
        print(test_setups, file=fout)
        for case in test_cases:
            print(case, file=fout)
        fout.close()
    command = Command(f"python {tempdir.name}/code-{i}.py >/dev/null 2>&1")
    execution_result = command.run(timeout=timeout) == 0
    return (task_id, execution_result)
    # return execution_result


""" dataset keys: src, trg_prediction, reference (only trg_prediction useful) """

def save_dict_as_json(d, filename):
    with open(f'{filename}.json', 'w') as fp:
        json.dump(d, fp)


def evaluate_google_mbpp(
    dataset,
    reference_path,
    split="test",
    timeout=10,
    num_procs=1,
    verbose=False,
):
    references = MBPPGoogleDataset(reference_path)
    tempdir = tempfile.TemporaryDirectory()
    passed_information = list()
    passed_information = collections.defaultdict(list)
    partial_evalutate_one = partial(
        evaluate_one_mbpp, tempdir=tempdir, references=references.raw_data[split][:len(dataset)], timeout=timeout
    )
    list_of_results = []

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            for result_json in tqdm(
                pool.imap(
                    partial_evalutate_one, list(enumerate(dataset))
                ),
                total=len(dataset),
                leave=False,
                disable=not verbose,
            ):
                passed_information[result_json[0]].append(result_json[1])
                # list_of_results.append(result_json)
    else:
        for args in tqdm(
            list(enumerate(dataset)), disable=not verbose
        ):
            result_json = partial_evalutate_one(args)
            list_of_results.append(result_json)
            passed_information[result_json[0]].append(result_json[1])
    tempdir.cleanup()

    save_dict_as_json(passed_information, "passed_information_100")
    save_dict_as_json({"results":list_of_results}, "list_of_results_100")

    results = {task_id: {"num_samples": len(sample_passed_info), "num_correct": sum(sample_passed_info)} for task_id, sample_passed_info in passed_information.items()}
    return results

def postprocess(datapoint):
    prediction = datapoint.get("trg_prediction")
    prediction = prediction.split("[INST]")[0].strip()
    prediction = prediction.split("if __name__")[0].strip()
    prediction = prediction.split("assert")[0].strip()
    prediction = "def ".join(prediction.split("def ")[:2]).strip()
    datapoint["trg_prediction"] = prediction

    return datapoint


def extract_first_function_refined(python_code):
    # Adjusted regular expression pattern
    # Stops capturing at two consecutive line breaks or the start of another 'def' or 'class'
    pattern = r'\bdef\s+\w+\s*\([^)]*\)\s*:\s*(.*?(?=(\n\s*\n)|^\b(def|class)\b))'
    match = re.search(pattern, python_code, re.DOTALL | re.MULTILINE)

    if match:
        return match.group()
    else:
        return None
    
def extract_first_code_block(text):
    # Regular expression pattern for code block enclosed in triple backticks
    pattern = r'```(.*?)```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()  # Removing leading and trailing whitespace
    else:
        return None

def post_process_codellama(datapoint):
    prediction = datapoint.get("trg_prediction")
    prediction = prediction.split("[/INST]")[1].strip()
    
    p = extract_first_code_block(prediction)
    if p is not None:
        prediction = p
    # if '```' in prediction:
    #     prediction = prediction.split('```')[1]

    # refined =extract_first_function_refined(prediction)
    # if refined is not None:
    #     prediction = refined
    
    datapoint["trg_prediction"] = prediction
    return datapoint
def postprocess_gpt_pseudocode(datapoint):
    prediction = datapoint.get("trg_prediction")
    prediction.split("ORIG")

    datapoint["trg_prediction"] = prediction

def extract_first_function(code_string):
    """
    Extracts the first complete function definition from a given string.
    
    Args:
    code_string (str): A string containing Python code with one or more function definitions.
    
    Returns:
    str: The first complete function definition, if found, otherwise an empty string.
    """
    # Regular expression pattern to match a function definition
    # This pattern looks for 'def' followed by valid function name and parameters,
    # and then captures everything until it reaches a line that is less indented 
    # than the function definition.
    function_pattern = r"(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\):(?:\n\s+.+)*)"

    # Search for the first function using the regular expression
    match = re.search(function_pattern, code_string, re.MULTILINE | re.DOTALL)

    # Return the matched function or an empty string if no match is found
    return match.group(0) if match else ""


def eval_mbpp(
        reference_path="data/mbpp.jsonl",
        num_procs=8,
        ):
    dataset = []
    # for filepath in glob("{}/*.jsonl".format(prediction_dir)):

    # !TODO: clean this up and make it more modular i.e that it can automatically detect the model name and path


    # generate the mbpp file/results


    parser = argparse.ArgumentParser(description="Eval model")
    # parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument('--model_name', type=str, default='openai/gpt3.5-turbo')
    parser.add_argument('--result_path', type=str, default="./results/codellama/mbpp_CodeLlama-7b-Instruct-hf_100.jsonl")
    # parser.add_argument('--result_path', type=str, default="./results/openai/mbpp_gpt3.5-turbo_100.jsonl")
    # parser.add_argument('--result_path', type=str, default="./results/openai/mbpp_oneshot_steps_gpt3.5-turbo_100.jsonl")

    # parser.add_argument('--result_path', type=str, default="./results/codellama/mbpp_CodeLlama-34b-Instruct-hf_100.jsonl")
    args = parser.parse_args()
    print(args)

    # out_path = "results/" + args.model_name.split('/')[0] + "/mbpp_" + args.model_name.split('/')[1] + '_' + str(args.length) + ".jsonl"    
    with open(args.result_path) as f:
        dataset.extend([json.loads(line) for line in f])

    dataset = list(map(postprocess, dataset)) #default postprocessing
    
    # dataset = list(map(post_process_codellama, dataset)) #ADDED FOR CODELLAMA 
    # save jsonl file
    out_path = "results/" + args.model_name.split('/')[0] + "/mbpp_" + "oneshot_modified_"+ args.model_name.split('/')[1] + '_' + str(100) + ".jsonl"
    with open(out_path, "w") as f:
        for item in dataset:
            json.dump(item, f)
            f.write("\n")
    f.close() 
    
    # statistics
    stats = collections.Counter([ex["task_id"] for ex in dataset])
    print(stats)

    results = evaluate_google_mbpp(dataset,
                                 reference_path,
                                 num_procs=num_procs,
                                 verbose=True)
    score_dict = collections.defaultdict(float)
    num_samples = [r["num_samples"] for r in results.values()]
    num_correct = [r["num_correct"] for r in results.values()]
    for k in (1, 3, 5): # Pass@1, Pass@3, Pass@5
        scores = estimate_pass_at_k(num_samples, num_correct, k)
        score_dict[k] = float(np.mean(scores))

    print("Results:\n")
    for k, score in score_dict.items():
        print(f"Pass@{k} = {score*100:.2f}%\n")
    
    print("\n")
    print(args)
    print(dataset[0]["trg_prediction"])

if __name__ == "__main__":
    fire.Fire(eval_mbpp)
