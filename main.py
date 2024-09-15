import argparse
from benchmark_runners import gsm_eval, mmlu_eval, human_ev_eval, bigbench_hard_eval, hellaswag_eval, drop_eval
import yaml

from types import SimpleNamespace

def dict_to_namespace(d):
    """Convert dict to SimpleNamespace recursively."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def run_benchmarks(config={}):
    results = []

    if config.eval_type == 'gsm':
        config = config.gsm
        result_gsm = gsm_eval.run_gsm_benchmark(config)

    elif config.eval_type == 'mmlu':
        config = config.mmlu
        result_mmlu = mmlu_eval.run_mmlu_benchmark(config)

    elif config.eval_type == 'human_eval':
        config = config.human_eval
        result_human_eval = human_eval.run_humaneval_benchmark(config)

    elif config.eval_type == 'mbpp':
        config = config.mbpp
        results_mbpp = mbpp_eval.run_benchmark(config)

    elif config.eval_type == 'bigbench_hard':
        config = config.bigbench_hard
        results_bigbench_hard = bigbench_hard_eval.run_bigbench_benchmark(
            config)

    elif config.eval_type == 'hellaswag':
        results_hellaswag = hellaswag_eval.run_hellaswag_benchmark(config)

    elif config.eval_type == 'drop':
        results_drop = drop_eval.run_drop_benchmark(config)

    return results


def parse_main_args():

    with open('config.yaml', 'r') as yaml_content:
        parser = argparse.ArgumentParser()
        parser.add_argument('--eval_type',
                            type=str,
                            default='gsm',
                            help='Evaluation type')

        config = yaml.safe_load(yaml_content)

    config = dict_to_namespace(config)  # convert to dot notation

    # Add more common parameters if needed
    return config, parser.parse_args()

if __name__ == "__main__":
    config, _ = parse_main_args()
    results = run_benchmarks(config)
    for result in results:
        print(result)
