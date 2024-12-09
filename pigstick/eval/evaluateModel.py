import os
import csv
from enum import Enum
from typing import Union

from datasets import load_dataset

class EvaluationMethod(Enum):
    UNIT_TEST = "unit test"
    GOLD_CODE = "gold code"

    @staticmethod
    def from_str(label):
        if label.lower() in ("unit test", "test", "unit_test"):
            return EvaluationMethod.UNIT_TEST
        elif label.lower() in ("gold code", "code", "gold_code",):
            return EvaluationMethod.GOLD_CODE
        else:
            print(f"[ERROR] {label} not a viable option")
            raise NotImplementedError(f"{label} not a supported method")

class Evaluation:
    def __init__(self, dataset_file: str) -> None:
        self.eval_data = []

        if not os.path.isfile(dataset_file):
            print(f"[ERROR] Dataset file {dataset_file} not found")
            raise ValueError(f"Dataset file {dataset_file} does not exist")

        print(f"[INFO] Retrieving evaluation data from datasets file {dataset_file}")
        with open(dataset_file, "r") as input_file:
            header = []
            reader = csv.reader(input_file)
            for header_line in reader:
                header = header_line
            header = [value.strip() for value in header]

            for line in reader:
                current_dataset = line.strip()
                print(f"[INFO] Loading dataset {current_dataset}")

                # dataset dictionary
                dataset_dict = {
                    "prompt": line[header.index("prompt")],
                    "code": line[header.index("code")],
                    "unit test values": line[header.index("unit_test_values")],
                    "dataset": load_dataset(line[header.index("dataset")])
                }
                self.eval_data.append(dataset_dict)

    def evaluate(self, model, method: Union[str, EvaluationMethod] = EvaluationMethod.UNIT_TEST, num_eval_points: int = -1, randomize: bool = True) -> float:
        method = method if isinstance(method, EvaluationMethod) else EvaluationMethod.from_str(method)
        if method != EvaluationMethod.UNIT_TEST and method != EvaluationMethod.GOLD_CODE:
            print(f"[ERROR] Evaluation method {method} not supported")
            raise ValueError(f"Evaluation method must be UNIT_TEST or GOLD_CODE, not {method}")
        if num_eval_points == -1:
            num_eval_points = float("inf")

        num_correct_points = 0
        num_total_points = 0
        for dataset in self.eval_data:
            # get prompt, code, & unit test values
            code = model.generate()
            if method == EvaluationMethod.UNIT_TEST:
                num_correct_points += self.eval_unit_test(code)
                num_total_points += 1
            elif method == EvaluationMethod.GOLD_CODE:
                num_correct_points += self.eval_gold_code(code )
                num_total_points += 1
        
        return num_correct_points / num_total_points

    def eval_unit_test(self, code: str, unit_test_values: list) -> int:
        return

    def eval_gold_code(self, code: str, gold_code_value: str) -> int:
        return 1 if code == gold_code_value else 0