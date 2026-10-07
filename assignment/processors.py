'''
Module processors
Implements base and derived dataset processors for training, ideal, and test datasets.
Follows Unit 2.2 of the course book for object-oriented design and inheritance.
'''

import math
import os
import pandas as pd

from assignment.exceptions import (
    DataLoadingError,
    DataValidationError,
    FunctionMatchingError,
)


def resolve_data_path(filepath):
    '''
    Resolves data file path across project root, notebooks, and subdirectories.
    '''
    if not filepath:
        return None
    if os.path.isabs(filepath) and os.path.exists(filepath):
        return filepath
    if os.path.exists(filepath):
        return filepath

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.basename(filepath)
    candidates = [
        os.path.join(base_dir, "data", filename),
        os.path.join(base_dir, filename),
        os.path.join("..", "data", filename),
        os.path.join("data", filename),
        os.path.join("..", filename),
        filepath
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return filepath


class BaseDatasetProcessor(object):
    '''
    Base class for dataset loading and validation.
    Provides shared attributes and methods for all dataset processors.
    '''

    def __init__(self, filepath=None):
        '''
        Constructor for BaseDatasetProcessor.
        filepath: path to CSV dataset file.
        '''
        self.filepath = resolve_data_path(filepath) if filepath else None
        self.data = None

    def load_data(self):
        '''
        Loads data from CSV file into a pandas DataFrame.
        Handles standard FileNotFoundError and raises DataLoadingError if missing.
        '''
        if self.filepath:
            self.filepath = resolve_data_path(self.filepath)

        if not self.filepath or not os.path.exists(self.filepath):
            raise DataLoadingError(self.filepath, f"File does not exist: {self.filepath}")

        try:
            self.data = pd.read_csv(self.filepath)
        except Exception as error:
            raise DataLoadingError(self.filepath, f"Could not read CSV file: {str(error)}")

        self.validate_data()
        return self.data

    def validate_data(self):
        '''
        Validates basic dataset integrity. Subclasses expand this method.
        '''
        if self.data is None or self.data.empty:
            raise DataValidationError(str(self.filepath), "Dataset is empty.")

    def get_dataframe(self):
        '''
        Returns loaded DataFrame.
        '''
        return self.data


class TrainingDatasetProcessor(BaseDatasetProcessor):
    '''
    Processor for 4 training datasets.
    Inherits from BaseDatasetProcessor.
    '''

    def __init__(self, filepath=None):
        '''
        Constructor for TrainingDatasetProcessor.
        '''
        if filepath is None:
            filepath = resolve_data_path("train.csv")
        super().__init__(filepath)
        self.expected_columns = ["x", "y1", "y2", "y3", "y4"]

    def validate_data(self):
        '''
        Validates column structure and numeric types for training data.
        '''
        super().validate_data()
        for col in self.expected_columns:
            if col not in self.data.columns:
                raise DataValidationError("TrainingData", f"Missing expected column: {col}")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise DataValidationError("TrainingData", f"Column {col} is not numeric.")

    def select_best_ideal_functions(self, ideal_processor):
        '''
        Selects the four ideal functions that best fit the training datasets.
        The criterion is minimizing the sum of all squared y deviations (Least-Squares).
        Also calculates the maximum deviation for each selected training and ideal pair.

        ideal_processor: an instantiated and loaded IdealDatasetProcessor object.
        return: dictionary containing chosen ideal function details for each training function.
        '''
        if self.data is None:
            self.load_data()

        ideal_df = ideal_processor.get_dataframe()
        if ideal_df is None:
            raise FunctionMatchingError("Ideal functions dataset is not loaded.")

        training_functions = ["y1", "y2", "y3", "y4"]
        ideal_function_columns = [f"y{i}" for i in range(1, 51)]

        chosen_functions = {}

        for t_col in training_functions:
            best_ideal_col = None
            lowest_squared_sum = float("inf")
            max_deviation_at_best = float("inf")

            for i_col in ideal_function_columns:
                deviation_series = self.data[t_col] - ideal_df[i_col]
                squared_sum = (deviation_series ** 2).sum()

                if squared_sum < lowest_squared_sum:
                    lowest_squared_sum = squared_sum
                    best_ideal_col = i_col
                    max_deviation_at_best = deviation_series.abs().max()

            if best_ideal_col is None:
                raise FunctionMatchingError(f"Failed to find ideal match for {t_col}.")

            threshold = max_deviation_at_best * math.sqrt(2)

            chosen_functions[t_col] = {
                "training_col": t_col,
                "chosen_ideal_col": best_ideal_col,
                "sum_squared_deviation": float(lowest_squared_sum),
                "max_deviation": float(max_deviation_at_best),
                "threshold": float(threshold)
            }

        return chosen_functions

    def get_regression_summary_table(self, chosen_functions):
        '''
        Compiles a structured summary table of the selected regression models.
        Useful for reporting and tabulating results.

        chosen_functions: dictionary returned by select_best_ideal_functions.
        return: list of dictionaries representing the regression summary table.
        '''
        summary = []
        for t_col, info in chosen_functions.items():
            summary.append({
                "training_function": t_col,
                "ideal_function": str(info["chosen_ideal_col"]).upper(),
                "sum_squared_error": float(info["sum_squared_deviation"]),
                "max_deviation": float(info["max_deviation"]),
                "threshold": float(info["threshold"])
            })
        return summary


class IdealDatasetProcessor(BaseDatasetProcessor):
    '''
    Processor for the fifty ideal functions.
    Inherits from BaseDatasetProcessor.
    '''

    def __init__(self, filepath=None):
        '''
        Constructor for IdealDatasetProcessor.
        '''
        if filepath is None:
            filepath = resolve_data_path("ideal.csv")
        super().__init__(filepath)

    def validate_data(self):
        '''
        Validates that fifty-one columns (x and y1 through y50) are present and numeric.
        '''
        super().validate_data()
        expected = ["x"] + [f"y{i}" for i in range(1, 51)]
        for col in expected:
            if col not in self.data.columns:
                raise DataValidationError("IdealFunctions", f"Missing expected column: {col}")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise DataValidationError("IdealFunctions", f"Column {col} is not numeric.")

    def get_ideal_values_by_x(self, x_value, ideal_columns):
        '''
        Retrieves y values for specified ideal columns at a specific x coordinate.
        x_value: float coordinate to look up.
        ideal_columns: list of ideal function column names.
        return: dictionary mapping ideal column name to its y value at x.
        '''
        matches = self.data[self.data["x"].round(4) == round(x_value, 4)]
        if matches.empty:
            return None
        return {col: float(matches[col].iloc[0]) for col in ideal_columns}


class TestDatasetProcessor(BaseDatasetProcessor):
    '''
    Processor for test data loaded line-by-line.
    Inherits from BaseDatasetProcessor.
    '''

    def __init__(self, filepath=None):
        '''
        Constructor for TestDatasetProcessor.
        '''
        if filepath is None:
            filepath = resolve_data_path("test.csv")
        super().__init__(filepath)
        self.expected_columns = ["x", "y"]

    def validate_data(self):
        '''
        Validates column structure and numeric types for test data.
        '''
        super().validate_data()
        for col in self.expected_columns:
            if col not in self.data.columns:
                raise DataValidationError("TestData", f"Missing expected column: {col}")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise DataValidationError("TestData", f"Column {col} is not numeric.")

    def stream_test_data(self):
        '''
        Generator yielding test data points line-by-line directly from the CSV file.
        Implements Unit 1.4 (Input/Output) and Unit 2.3 (Generators and Iterators).
        yields: tuple of (x, y) float values for each line.
        '''
        if self.data is not None:
            for _, row in self.data.iterrows():
                yield float(row["x"]), float(row["y"])
            return

        if self.filepath:
            self.filepath = resolve_data_path(self.filepath)

        if not self.filepath or not os.path.exists(self.filepath):
            raise DataLoadingError(self.filepath, f"File does not exist: {self.filepath}")

        try:
            with open(self.filepath, "r", encoding="utf-8") as file:
                header = file.readline()
                if not header:
                    raise DataValidationError("TestData", "Empty test CSV file.")

                line_number = 1
                for line in file:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        raise DataValidationError("TestData", f"Malformed row at line {line_number}: {line}")
                    try:
                        x_val = float(parts[0].strip())
                        y_val = float(parts[1].strip())
                    except ValueError as error:
                        raise DataValidationError("TestData", f"Non-numeric value at line {line_number}: {str(error)}")
                    yield x_val, y_val
        except AssignmentBaseException as error:
            raise error
        except Exception as error:
            raise DataLoadingError(self.filepath, f"Error streaming test data: {str(error)}")

    def map_test_data(self, ideal_processor, chosen_models):
        '''
        Processes test data line-by-line and maps each point to the best chosen ideal function.
        A test point is mapped if its deviation from the ideal function does not exceed:
        max_deviation_between_training_and_ideal * sqrt(2).
        If multiple chosen functions qualify, the one with minimal deviation is selected.

        ideal_processor: IdealDatasetProcessor object.
        chosen_models: dictionary containing chosen ideal function criteria.
        return: tuple of (mapped_records, all_records_summary).
        '''
        mapped_records = []
        all_records_summary = []

        active_ideal_cols = [info["chosen_ideal_col"] for info in chosen_models.values()]

        # Stream line-by-line via generator protocol (Unit 2.3)
        for test_x, test_y in self.stream_test_data():
            ideal_vals = ideal_processor.get_ideal_values_by_x(test_x, active_ideal_cols)
            if ideal_vals is None:
                continue

            best_match_col = None
            smallest_deviation = float("inf")

            for t_col, info in chosen_models.items():
                i_col = info["chosen_ideal_col"]
                ideal_y = ideal_vals[i_col]
                deviation = abs(test_y - ideal_y)
                threshold = info["threshold"]

                if deviation <= threshold:
                    if deviation < smallest_deviation:
                        smallest_deviation = deviation
                        best_match_col = i_col

            if best_match_col is not None:
                ideal_identifier = str(best_match_col).upper()
                record = {
                    "X (test func)": test_x,
                    "Y (test func)": test_y,
                    "Delta Y (test func)": float(smallest_deviation),
                    "No. of ideal func": ideal_identifier
                }
                mapped_records.append(record)
                all_records_summary.append({
                    "x": test_x,
                    "y": test_y,
                    "delta_y": float(smallest_deviation),
                    "ideal_func": ideal_identifier,
                    "mapped": True
                })
            else:
                all_records_summary.append({
                    "x": test_x,
                    "y": test_y,
                    "delta_y": None,
                    "ideal_func": None,
                    "mapped": False
                })

        return mapped_records, all_records_summary
