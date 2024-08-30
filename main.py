import argparse
from benchmarks import gsm_eval, mmlu_eval, human_eval

def run_all_benchmarks(config):
    results = []

    if config.eval_type == 'gsm':
        result_gsm = gsm_eval.run_benchmark(config)
        results.append(result_gsm)
    
    elif config.eval_type == 'mmlu':
        result_mmlu = mmlu_eval.run_benchmark(config)
        results.append(result_mmlu)
    
    elif config.eval_type == 'deepemo':
        result_deepemo = deepemo.run_benchmark(config)
        results.append(result_deepemo)
    
    return results

def parse_main_args():
    parser = argparse.ArgumentParser(description="Main Handler for Evaluations")
    parser.add_argument('--eval_type', type=str, required=True, choices=['gsm', 'mmlu', 'deepemo'], help='Type of evaluation to run')
    parser.add_argument('--param1', type=str, help='Parameter 1 for the chosen evaluation')
    parser.add_argument('--param2', type=str, help='Parameter 2 for the chosen evaluation')
    # Add more common parameters if needed
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_main_args()
    results = run_all_benchmarks(config)
    for result in results:
        print(result)
