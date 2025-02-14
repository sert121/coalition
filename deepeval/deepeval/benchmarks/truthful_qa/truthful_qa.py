from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import re

from deepeval.dataset import Golden
from deepeval.benchmarks.base_benchmark import DeepEvalBaseBenchmark
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
from deepeval.benchmarks.utils import should_use_batch
from deepeval.scorer import Scorer


class TruthfulQA(DeepEvalBaseBenchmark):

    def extract_and_parse_list(self,text):
        # Define the regex pattern to capture the first list
        pattern = r'\[(.*?)\]'

        # Find the first match
        match = re.search(pattern, text)

        if match:
            # Extract the content inside the list
            list_content = match.group(1)

            # Split the content by comma and strip any whitespace
            parsed_list = [item.strip() for item in list_content.split(',')]

            return parsed_list
        else:
            return None

    def __init__(
        self,
        tasks: List[TruthfulQATask] = None,
        mode: TruthfulQAMode = TruthfulQAMode.MC1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tasks: List[TruthfulQATask] = (list(TruthfulQATask)
                                            if tasks is None else tasks)
        self.mode: TruthfulQAMode = mode
        self.scorer = Scorer()
        self.mc_dataset: Dataset = self.dataset

        self.predictions: Optional[pd.DataFrame] = None
        self.task_scores: Optional[pd.DataFrame] = None
        self.overall_score: Optional[float] = None

    def evaluate(self,
                 model: DeepEvalBaseLLM,
                 batch_size: Optional[int] = None) -> Dict:
        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        scores_row = []
        use_batch = should_use_batch(model, batch_size)

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task, self.mode)
            task_correct_predictions = 0
            task_total_predictions = len(goldens)
            overall_total_predictions += len(goldens)

            # Calculate task accuracy
            if use_batch:
                for i in tqdm(
                        range(0, len(goldens), batch_size),
                        desc=
                        f"Batch Processing {task.value} (batch_size={batch_size})",
                ):
                    goldens_batch = goldens[i:i + batch_size]
                    batch_predictions = self.batch_predict(
                        model, goldens_batch, self.mode)
                    for golden, prediction_dict in zip(goldens_batch,
                                                       batch_predictions):
                        prediction = prediction_dict["prediction"]
                        score = prediction_dict["score"]
                        if score:
                            task_correct_predictions += 1
                            overall_correct_predictions += 1
                        predictions_row.append(
                            (task.value, golden.input, prediction, score))
            else:
                for golden in tqdm(goldens, desc=f"Processing {task.value}"):
                    prediction, score = self.predict(model, golden,
                                                     self.mode).values()
                    if score:
                        task_correct_predictions += score
                        overall_correct_predictions += score
                    predictions_row.append(
                        (task.value, golden.input, prediction, score))

            task_accuracy = task_correct_predictions / task_total_predictions
            print(
                f"TruthfulQA Task Accuracy (task={task.value}): {task_accuracy}"
            )
            scores_row.append((task.value, task_accuracy))

        # Calculate overall accuracy
        overall_accuracy = (overall_correct_predictions /
                            overall_total_predictions)
        print(f"Overall TruthfulQA Accuracy: {overall_accuracy}")

        # Create a DataFrame from task_results_data
        # Columns: 'Task', 'Input', 'Prediction', 'Score'
        self.predictions = pd.DataFrame(
            predictions_row,
            columns=["Task", "Input", "Prediction", "Correct"])
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy

    def predict(self, model: DeepEvalBaseLLM, golden: Golden,
                mode: TruthfulQAMode) -> Dict:
        # Define prompt template
        prompt: dict = TruthfulQATemplate.generate_output(input=golden.input,
                                                          mode=mode)
        prediction = model.generate(prompt)

        if isinstance(prediction, str):
            extracted_list = self.extract_and_parse_list(prediction)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Define Metric
        if mode == TruthfulQAMode.MC1:
            print("** the golden expected output is: ", golden.expected_output)
            try:
                prediction = extracted_list[0] # parsing the actual multi choice answer
            except:
                pass
            score = self.scorer.exact_match_score(golden.expected_output,
                                                  prediction)
        elif mode == TruthfulQAMode.MC2:

            print("** the golden expected output is: ", golden.expected_output)
            try:
                prediction = extracted_list[0] # parsing the actual multi choice answer
            except:
                pass
            score = self.scorer.truth_identification_score(
                golden.expected_output, prediction)

        return {"prediction": prediction, "score": score}

    def batch_predict(
        self,
        model: DeepEvalBaseLLM,
        goldens: List[Golden],
        mode: TruthfulQAMode,
    ) -> List[Dict]:
        # Define prompt template
        prompts = []
        for golden in goldens:
            prompt: dict = TruthfulQATemplate.generate_output(
                input=golden.input, mode=mode)
            prompts.append(prompt)
        predictions = model.batch_generate(prompts)
        if len(predictions) is not len(goldens):
            raise ValueError(
                "Custom `batch_generate` method did not return the same number of generations as the number of prompts."
            )

        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            golden = goldens[i]
            # Define Metric
            if mode == TruthfulQAMode.MC1:
                print("** the golden expected output is: ",
                      golden.expected_output)
                score = self.scorer.exact_match_score(golden.expected_output,
                                                      prediction)
            elif mode == TruthfulQAMode.MC2:
                score = self.scorer.truth_identification_score(
                    golden.expected_output, prediction)
            res.append({"prediction": prediction, "score": score})

        return res

    def load_benchmark_dataset(self, task: TruthfulQATask,
                               mode: TruthfulQAMode) -> List[Golden]:
        # Load full dataset
        if self.mc_dataset is None:
            gen_dataset = load_dataset("truthful_qa",
                                       "generation")["validation"]
            mc_dataset = load_dataset("truthful_qa",
                                      "multiple_choice")["validation"]
            df_mc, df_gen = mc_dataset.to_pandas(), gen_dataset.to_pandas()
            merged_df = pd.merge(
                df_mc,
                df_gen[["question", "category"]],
                on="question",
                how="left",
            )
            mc_dataset = Dataset.from_pandas(merged_df)
            self.mc_dataset = mc_dataset
        else:
            mc_dataset = self.mc_dataset
        dataset = self.mc_dataset.filter(
            lambda data: data["category"] == task.value)

        # Create goldens list from datset
        goldens: List[Golden] = []
        for data in dataset:
            if mode == TruthfulQAMode.MC1:
                input, expected_output = TruthfulQATemplate.format_mc1_question(
                    data)
                golden = Golden(input=input, expected_output=expected_output)
                goldens.append(golden)
            elif mode == TruthfulQAMode.MC2:
                input, expected_output = TruthfulQATemplate.format_mc2_question(
                    data)
                golden = Golden(input=input,
                                expected_output=str(expected_output))
                goldens.append(golden)

        return goldens
