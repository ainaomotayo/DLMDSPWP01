'''
Unit Test Suite for DLMDSPWP01 Assignment
IU International University of Applied Sciences

Tests all core functional elements:
- Custom exceptions hierarchy
- Base and derived dataset loaders and validators
- Least-Squares fitting and threshold calculations
- Line-by-line test data classification criterion
- Database schema compilation and CRUD operations
Follows Unit 5.3 of the course book.
'''

import math
import os
import sys
import unittest
import pandas as pd
import sqlalchemy as db

# Ensure project root is in sys.path when executing test script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from assignment import (
    AssignmentBaseException,
    DataLoadingError,
    DataValidationError,
    FunctionMatchingError,
    DatabaseOperationError,
    TrainingDatasetProcessor,
    IdealDatasetProcessor,
    TestDatasetProcessor,
    DatabaseManager,
)


class TestCustomExceptions(unittest.TestCase):
    '''
    Tests inheritance and behavior of user-defined exceptions.
    '''

    def test_exception_inheritance(self):
        '''
        Checks that custom exceptions inherit properly from base classes.
        '''
        base_err = AssignmentBaseException("base error")
        self.assertIsInstance(base_err, Exception)

        load_err = DataLoadingError("sample.csv", "file missing")
        self.assertIsInstance(load_err, AssignmentBaseException)
        self.assertIsInstance(load_err, Exception)

        val_err = DataValidationError("TestDataset", "missing column")
        self.assertIsInstance(val_err, AssignmentBaseException)

        match_err = FunctionMatchingError("matching error")
        self.assertIsInstance(match_err, AssignmentBaseException)

        db_err = DatabaseOperationError("insert", "failed")
        self.assertIsInstance(db_err, AssignmentBaseException)


class TestDatasetLoaders(unittest.TestCase):
    '''
    Tests data ingestion and schema validation for datasets.
    '''

    def setUp(self):
        '''
        Prepares temporary test CSV files.
        '''
        self.temp_csv = "temp_valid.csv"
        self.empty_csv = "temp_empty.csv"
        self.bad_csv = "temp_bad.csv"

        valid_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "y1": [10.0, 20.0],
            "y2": [15.0, 25.0],
            "y3": [20.0, 30.0],
            "y4": [25.0, 35.0]
        })
        valid_df.to_csv(self.temp_csv, index=False)

        empty_df = pd.DataFrame()
        empty_df.to_csv(self.empty_csv, index=False)

        bad_df = pd.DataFrame({"x": [1.0, 2.0], "y1": [10.0, 20.0]})
        bad_df.to_csv(self.bad_csv, index=False)

    def tearDown(self):
        '''
        Cleans up temporary CSV files.
        '''
        for path in [self.temp_csv, self.empty_csv, self.bad_csv]:
            if os.path.exists(path):
                os.remove(path)

    def test_missing_file_raises_data_loading_error(self):
        '''
        Checks that missing CSV files raise DataLoadingError.
        '''
        processor = TrainingDatasetProcessor("non_existent_file.csv")
        with self.assertRaises(DataLoadingError):
            processor.load_data()

    def test_valid_training_data_loading(self):
        '''
        Checks that valid training CSV loads properly into a DataFrame.
        '''
        processor = TrainingDatasetProcessor(self.temp_csv)
        df = processor.load_data()
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["x", "y1", "y2", "y3", "y4"])

    def test_missing_columns_raises_data_validation_error(self):
        '''
        Checks that training dataset with missing columns raises DataValidationError.
        '''
        processor = TrainingDatasetProcessor(self.bad_csv)
        with self.assertRaises(DataValidationError):
            processor.load_data()

    def test_empty_dataset_raises_data_validation_error(self):
        '''
        Checks that an empty file raises DataLoadingError or DataValidationError.
        '''
        processor = TrainingDatasetProcessor(self.empty_csv)
        with self.assertRaises(AssignmentBaseException):
            processor.load_data()


class TestFunctionMatchingLogic(unittest.TestCase):
    '''
    Tests Least-Squares regression selection and threshold computation.
    '''

    def test_least_squares_exact_synthetic_match(self):
        '''
        Verifies that Least-Squares correctly identifies an identical function.
        '''
        train_processor = TrainingDatasetProcessor()
        train_processor.data = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, 2.0, 3.0],
            "y2": [10.0, 20.0, 30.0],
            "y3": [5.0, 5.0, 5.0],
            "y4": [0.0, 1.0, 4.0]
        })

        ideal_cols = {"x": [0.0, 1.0, 2.0]}
        for i in range(1, 51):
            if i == 5:
                ideal_cols["y5"] = [1.0, 2.0, 3.0]
            elif i == 10:
                ideal_cols["y10"] = [10.0, 20.0, 30.0]
            elif i == 15:
                ideal_cols["y15"] = [5.0, 5.0, 5.0]
            elif i == 20:
                ideal_cols["y20"] = [0.0, 1.0, 4.0]
            else:
                ideal_cols[f"y{i}"] = [100.0, 100.0, 100.0]

        ideal_processor = IdealDatasetProcessor()
        ideal_processor.data = pd.DataFrame(ideal_cols)

        chosen = train_processor.select_best_ideal_functions(ideal_processor)
        self.assertEqual(chosen["y1"]["chosen_ideal_col"], "y5")
        self.assertEqual(chosen["y2"]["chosen_ideal_col"], "y10")
        self.assertEqual(chosen["y3"]["chosen_ideal_col"], "y15")
        self.assertEqual(chosen["y4"]["chosen_ideal_col"], "y20")

        self.assertAlmostEqual(chosen["y1"]["sum_squared_deviation"], 0.0)
        self.assertAlmostEqual(chosen["y1"]["max_deviation"], 0.0)
        self.assertAlmostEqual(chosen["y1"]["threshold"], 0.0)

    def test_actual_dataset_matches(self):
        '''
        Verifies that actual assignment datasets match expected ideal functions:
        y1 -> y13, y2 -> y24, y3 -> y36, y4 -> y40.
        '''
        train_p = TrainingDatasetProcessor()
        ideal_p = IdealDatasetProcessor()
        train_p.load_data()
        ideal_p.load_data()

        chosen = train_p.select_best_ideal_functions(ideal_p)
        self.assertEqual(chosen["y1"]["chosen_ideal_col"], "y13")
        self.assertEqual(chosen["y2"]["chosen_ideal_col"], "y24")
        self.assertEqual(chosen["y3"]["chosen_ideal_col"], "y36")
        self.assertEqual(chosen["y4"]["chosen_ideal_col"], "y40")

        for key in ["y1", "y2", "y3", "y4"]:
            info = chosen[key]
            self.assertGreater(info["max_deviation"], 0.0)
            self.assertAlmostEqual(info["threshold"], info["max_deviation"] * math.sqrt(2))

    def test_get_regression_summary_table(self):
        '''
        Verifies that get_regression_summary_table compiles structured model summary.
        '''
        train_p = TrainingDatasetProcessor()
        ideal_p = IdealDatasetProcessor()
        train_p.load_data()
        ideal_p.load_data()
        chosen = train_p.select_best_ideal_functions(ideal_p)
        summary = train_p.get_regression_summary_table(chosen)
        self.assertEqual(len(summary), 4)
        self.assertEqual(summary[0]["training_function"], "y1")
        self.assertEqual(summary[0]["ideal_function"], "Y13")


class TestTestDataMapping(unittest.TestCase):
    '''
    Tests line-by-line test data mapping criterion.
    '''

    def test_mapping_criterion_thresholds(self):
        '''
        Tests that points below threshold are accepted and above threshold are rejected.
        '''
        ideal_p = IdealDatasetProcessor()
        ideal_p.data = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y13": [10.0, 20.0, 30.0],
            "y24": [100.0, 200.0, 300.0],
            "y36": [1.0, 2.0, 3.0],
            "y40": [50.0, 60.0, 70.0]
        })

        chosen_models = {
            "y1": {"chosen_ideal_col": "y13", "threshold": 0.70},
            "y2": {"chosen_ideal_col": "y24", "threshold": 0.70},
            "y3": {"chosen_ideal_col": "y36", "threshold": 0.70},
            "y4": {"chosen_ideal_col": "y40", "threshold": 0.70}
        }

        test_p = TestDatasetProcessor()
        # Point 1: at x=1.0, y=10.5 -> dev=0.5 <= 0.7 (matches y13)
        # Point 2: at x=2.0, y=25.0 -> dev=5.0 > 0.7 (no match)
        test_p.data = pd.DataFrame({
            "x": [1.0, 2.0],
            "y": [10.5, 25.0]
        })

        mapped, summary = test_p.map_test_data(ideal_p, chosen_models)
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0]["No. of ideal func"], "Y13")
        self.assertAlmostEqual(mapped[0]["Delta Y (test func)"], 0.5)

        self.assertTrue(summary[0]["mapped"])
        self.assertFalse(summary[1]["mapped"])

    def test_actual_test_dataset_mapped_count(self):
        '''
        Verifies that 34 points out of 100 test records map to the four chosen functions.
        '''
        train_p = TrainingDatasetProcessor()
        ideal_p = IdealDatasetProcessor()
        test_p = TestDatasetProcessor()

        train_p.load_data()
        ideal_p.load_data()
        test_p.load_data()

        chosen = train_p.select_best_ideal_functions(ideal_p)
        mapped, summary = test_p.map_test_data(ideal_p, chosen)

        self.assertEqual(len(test_p.get_dataframe()), 100)
        self.assertEqual(len(mapped), 34)
        self.assertEqual(len(summary), 100)

    def test_stream_test_data_generator(self):
        '''
        Verifies line-by-line generator streaming directly from CSV (Unit 2.3).
        '''
        test_p = TestDatasetProcessor()
        streamed_points = list(test_p.stream_test_data())
        self.assertEqual(len(streamed_points), 100)
        first_x, first_y = streamed_points[0]
        self.assertEqual(first_x, -13.1)
        self.assertEqual(first_y, -4494.98)


class TestDatabaseManager(unittest.TestCase):
    '''
    Tests SQLite database creation and CRUD operations.
    '''

    def setUp(self):
        '''
        Creates a temporary test database file.
        '''
        self.test_db_file = "outputs/test_unit.db"
        self.db = DatabaseManager(self.test_db_file)
        self.db.compile_database_schema()

    def tearDown(self):
        '''
        Removes temporary database file.
        '''
        if os.path.exists(self.test_db_file):
            os.remove(self.test_db_file)

    def test_schema_tables_exist(self):
        '''
        Verifies that training_data, ideal_functions, and test_data_mapping tables are created.
        '''
        inspector = db.inspect(self.db.engine)
        table_names = inspector.get_table_names()
        self.assertIn("training_data", table_names)
        self.assertIn("ideal_functions", table_names)
        self.assertIn("test_data_mapping", table_names)

    def test_training_data_insert_and_count(self):
        '''
        Tests inserting training data records and querying count.
        '''
        df = pd.DataFrame({
            "x": [1.0, 2.0],
            "y1": [10.0, 20.0],
            "y2": [11.0, 21.0],
            "y3": [12.0, 22.0],
            "y4": [13.0, 23.0]
        })
        self.db.save_training_data(df)
        count = self.db.fetch_records_count("training_data")
        self.assertEqual(count, 2)

    def test_test_mapping_insert_and_count(self):
        '''
        Tests inserting mapped test records into Table 3.
        '''
        records = [
            {
                "X (test func)": 1.0,
                "Y (test func)": 10.2,
                "Delta Y (test func)": 0.2,
                "No. of ideal func": "Y13"
            }
        ]
        self.db.save_test_data_mapping(records)
        count = self.db.fetch_records_count("test_data_mapping")
        self.assertEqual(count, 1)

    def test_unknown_table_count_raises_database_error(self):
        '''
        Tests that querying an unknown table raises DatabaseOperationError.
        '''
        with self.assertRaises(DatabaseOperationError):
            self.db.fetch_records_count("non_existent_table")


class TestVisualizer(unittest.TestCase):
    '''
    Tests interactive Bokeh visual dashboard generation.
    '''

    def setUp(self):
        '''
        Prepares temporary HTML output path and minimal synthetic data.
        '''
        self.temp_html = "outputs/test_visualization.html"
        self.train_df = pd.DataFrame({
            "x": [1.0, 2.0],
            "y1": [10.0, 20.0],
            "y2": [15.0, 25.0],
            "y3": [20.0, 30.0],
            "y4": [25.0, 35.0]
        })
        ideal_data = {"x": [1.0, 2.0]}
        for i in range(1, 51):
            ideal_data[f"y{i}"] = [float(i), float(i * 2)]
        self.ideal_df = pd.DataFrame(ideal_data)

        self.chosen_models = {
            "y1": {"chosen_ideal_col": "y1", "max_deviation": 0.5, "threshold": 0.70},
            "y2": {"chosen_ideal_col": "y2", "max_deviation": 0.5, "threshold": 0.70},
            "y3": {"chosen_ideal_col": "y3", "max_deviation": 0.5, "threshold": 0.70},
            "y4": {"chosen_ideal_col": "y4", "max_deviation": 0.5, "threshold": 0.70}
        }
        self.all_summary = [
            {"x": 1.0, "y": 1.2, "delta_y": 0.2, "ideal_func": "Y1", "mapped": True},
            {"x": 2.0, "y": 50.0, "delta_y": None, "ideal_func": None, "mapped": False}
        ]

    def tearDown(self):
        '''
        Cleans up temporary HTML dashboard file.
        '''
        if os.path.exists(self.temp_html):
            os.remove(self.temp_html)

    def test_generate_visualization_creates_html(self):
        '''
        Verifies that DataVisualizer compiles and saves a valid HTML dashboard.
        '''
        from assignment import DataVisualizer
        visualizer = DataVisualizer(self.temp_html)
        result_path = visualizer.generate_visualization(
            self.train_df, self.ideal_df, self.chosen_models, self.all_summary
        )
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 1000)

    def test_export_png_creates_figures(self):
        '''
        Verifies that DataVisualizer exports static PNG figures for the report.
        '''
        from assignment import DataVisualizer
        visualizer = DataVisualizer()
        output_dir = "outputs/test_figures"
        files = visualizer.export_png(
            self.train_df, self.ideal_df, self.chosen_models, self.all_summary,
            output_dir=output_dir, combined_filename="test_combined.png"
        )
        self.assertEqual(len(files["individual"]), 5)
        self.assertTrue(os.path.exists(files["combined"]))
        for fpath in files["individual"]:
            self.assertTrue(os.path.exists(fpath))
            os.remove(fpath)
        os.remove(files["combined"])
        if os.path.exists(output_dir):
            os.rmdir(output_dir)


class TestValidationEdgeCases(unittest.TestCase):
    '''
    Tests validation edge cases across Ideal and Test processors.
    '''

    def test_ideal_processor_missing_columns_validation(self):
        '''
        Verifies that IdealDatasetProcessor raises DataValidationError when columns are missing.
        '''
        ideal_p = IdealDatasetProcessor()
        ideal_p.data = pd.DataFrame({"x": [1.0, 2.0], "y1": [1.0, 2.0]})
        with self.assertRaises(DataValidationError):
            ideal_p.validate_data()

    def test_test_processor_missing_columns_validation(self):
        '''
        Verifies that TestDatasetProcessor raises DataValidationError when columns are missing.
        '''
        test_p = TestDatasetProcessor()
        test_p.data = pd.DataFrame({"x": [1.0, 2.0]})
        with self.assertRaises(DataValidationError):
            test_p.validate_data()

    def test_get_ideal_values_by_x_unmatched_coordinate(self):
        '''
        Verifies that looking up an unmatched coordinate returns None.
        '''
        ideal_p = IdealDatasetProcessor()
        ideal_p.data = pd.DataFrame({"x": [1.0, 2.0], "y1": [10.0, 20.0]})
        result = ideal_p.get_ideal_values_by_x(99.0, ["y1"])
        self.assertIsNone(result)

    def test_unloaded_ideal_data_raises_function_matching_error(self):
        '''
        Verifies that selecting ideal functions without loaded ideal data raises FunctionMatchingError.
        '''
        train_p = TrainingDatasetProcessor()
        train_p.data = pd.DataFrame({
            "x": [1.0], "y1": [1.0], "y2": [2.0], "y3": [3.0], "y4": [4.0]
        })
        ideal_p = IdealDatasetProcessor()
        ideal_p.data = None
        with self.assertRaises(FunctionMatchingError):
            train_p.select_best_ideal_functions(ideal_p)


if __name__ == "__main__":
    unittest.main()
