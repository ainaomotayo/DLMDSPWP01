'''
Main client module for DLMDSPWP01 written assignment.
Coordinates dataset loading, ideal function selection, test data mapping,
database operations, and interactive visual generation.
Follows Unit 1.5 of the course book for modular package organization.
'''

import sys
from assignment import (
    AssignmentBaseException,
    TrainingDatasetProcessor,
    IdealDatasetProcessor,
    TestDatasetProcessor,
    DatabaseManager,
    DataVisualizer,
)


def main():
    '''
    Executes the complete assignment workflow:
    1. Loads datasets from CSV files.
    2. Initializes SQLite database and compiles schema.
    3. Identifies the four best ideal functions via Least-Squares criterion.
    4. Evaluates test data line-by-line and maps points meeting regression criterion.
    5. Stores datasets and test mapping in SQLite database tables.
    6. Produces interactive Bokeh visualization.
    '''
    print("=" * 70)
    print("DLMDSPWP01: Written Assignment Pipeline")
    print("IU International University of Applied Sciences")
    print("=" * 70)

    # 1. Dataset Processors Initialization
    train_processor = TrainingDatasetProcessor()
    ideal_processor = IdealDatasetProcessor()
    test_processor = TestDatasetProcessor()

    try:
        print("[1/5] Loading and validating CSV datasets...")
        train_df = train_processor.load_data()
        ideal_df = ideal_processor.load_data()
        test_df = test_processor.load_data()
        print(f"      Training data loaded: {len(train_df)} rows, 4 functions.")
        print(f"      Ideal functions loaded: {len(ideal_df)} rows, 50 functions.")
        print(f"      Test data loaded: {len(test_df)} rows.")
    except AssignmentBaseException as err:
        print(f"Error loading datasets: {err.message}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected error during data loading: {str(err)}", file=sys.stderr)
        return 1

    # 2. Database Initialization
    db_manager = DatabaseManager("outputs/assignment.db")
    try:
        print("[2/5] Compiling SQLite database schema via SQLAlchemy...")
        db_manager.compile_database_schema()
        print("      Database tables compiled: training_data, ideal_functions, test_data_mapping.")

        print("      Populating Table 1 (training data) and Table 2 (ideal functions)...")
        db_manager.save_training_data(train_df)
        db_manager.save_ideal_functions(ideal_df)
        print(f"      Table 1 records: {db_manager.fetch_records_count('training_data')}")
        print(f"      Table 2 records: {db_manager.fetch_records_count('ideal_functions')}")
    except AssignmentBaseException as err:
        print(f"Database error: {err.message}", file=sys.stderr)
        return 1

    # 3. Choose Four Ideal Functions via Least-Squares
    try:
        print("[3/5] Selecting best four ideal functions using Least-Squares criterion...")
        chosen_functions = train_processor.select_best_ideal_functions(ideal_processor)
        print("-" * 70)
        print(f"{'Training Func':<15}{'Chosen Ideal':<15}{'Sum Squared Dev':<20}{'Max Dev':<12}{'Threshold (sqrt(2))':<15}")
        print("-" * 70)
        for t_col, info in chosen_functions.items():
            print(f"{t_col:<15}{info['chosen_ideal_col']:<15}{info['sum_squared_deviation']:<20.4f}"
                  f"{info['max_deviation']:<12.4f}{info['threshold']:<15.4f}")
        print("-" * 70)
    except AssignmentBaseException as err:
        print(f"Function matching error: {err.message}", file=sys.stderr)
        return 1

    # 4. Map Test Data Line-by-Line
    try:
        print("[4/5] Processing test data line-by-line against criterion...")
        mapped_records, all_summary = test_processor.map_test_data(ideal_processor, chosen_functions)
        print(f"      Test points evaluated: {len(test_df)}")
        print(f"      Test points successfully assigned: {len(mapped_records)}")
        print(f"      Test points unassigned: {len(test_df) - len(mapped_records)}")

        print("      Saving mapped test results into Table 3 in SQLite database...")
        db_manager.save_test_data_mapping(mapped_records)
        print(f"      Table 3 records: {db_manager.fetch_records_count('test_data_mapping')}")
    except AssignmentBaseException as err:
        print(f"Test mapping error: {err.message}", file=sys.stderr)
        return 1

    # 5. Visualizations via Bokeh
    try:
        print("[5/5] Generating Bokeh interactive visualization...")
        visualizer = DataVisualizer("outputs/visualization.html")
        output_file_path = visualizer.generate_visualization(
            train_df, ideal_df, chosen_functions, all_summary
        )
        print(f"      Visualization saved successfully: {output_file_path}")
    except Exception as err:
        print(f"Visualization error: {str(err)}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Execution complete: All pipeline stages executed successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
