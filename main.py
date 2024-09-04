import argparse
from benchmark_runners import gsm_eval, mmlu_eval, human_ev_eval, bigbench_hard_eval, hellaswag_eval, drop_eval


def run_benchmarks(config={}):
    results = []

    if config.eval_type == 'gsm':
        result_gsm = gsm_eval.run_gsm_benchmark(config)

    elif config.eval_type == 'mmlu':
        result_mmlu = mmlu_eval.run_mmlu_benchmark(config)

    elif config.eval_type == 'human_eval':
        result_human_eval = human_eval.run_humaneval_benchmark(config)

    elif config.eval_type == 'mbpp':
        results_mbpp = mbpp_eval.run_benchmark(config)

    elif config.eval_type == 'bigbench_hard':
        results_bigbench_hard = bigbench_hard_eval.run_bigbench_benchmark(
            config)

    elif config.eval_type == 'hellaswag':
        results_hellaswag = hellaswag_eval.run_hellaswag_benchmark(config)

    elif config.eval_type == 'drop':
        results_drop = drop_eval.run_drop_benchmark(config)

    return results

def parse_main_args():
    parser = argparse.ArgumentParser(description="Main Handler for Evaluations")
    parser.add_argument('--eval_type', type=str, required=True, choices=['gsm', 'mmlu', 'human_eval'
    'mbpp', 'bigbench_hard', 'hellaswag', 'drop'], help='Type of evaluation to run')
    parser.add_argument('--param1', type=str, help='Parameter 1 for the chosen evaluation')
    parser.add_argument('--param2', type=str, help='Parameter 2 for the chosen evaluation')
    # Add more common parameters if needed
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_main_args()
    results = run_benchmarks(config)
    for result in results:
        print(result)
