# DLMDSPWP01: Programming with Python Written Assignment
IU International University of Applied Sciences

This repository contains the software implementation for the written assignment DLMDSPWP01. The application executes a scientific data analysis pipeline to evaluate training data, select the best fitting ideal functions using the Least-Squares criterion, map test data points line-by-line based on deviation thresholds, persist all datasets into an SQLite database via SQLAlchemy, and generate interactive visual dashboards using Bokeh.

## 1. Project Overview and Objectives

The primary tasks fulfilled by this application are:
- Ingest and validate four training datasets, fifty candidate ideal functions, and one test dataset.
- Select the four ideal functions that minimize the sum of squared deviations across all fifty candidate functions.
- Compute the maximum deviation for each selected function pair and establish individual mapping thresholds scaled by factor sqrt(2).
- Stream test data line-by-line from the CSV file and map qualifying points to the chosen ideal function with minimum deviation.
- Compile and populate an SQLite database with Table 1 (training data), Table 2 (ideal functions), and Table 3 (mapped test data).
- Generate an interactive multi-panel visualization with tolerance bands, scatter points, and hover tooltips.
- Validate all core logic, mathematical operations, generator streaming, and database persistence using automated unit tests.

## 2. Mathematical Methodology and Results

### Ideal Function Selection Criterion
For each training function k in {1, 2, 3, 4}, every ideal function j in {1, ..., 50} is evaluated to minimize the sum of squared errors (SSE):

$$\text{SSE}_{k, j} = \sum_{i=1}^{N} (y_{\text{train}, k}^{(i)} - y_{\text{ideal}, j}^{(i)})^2$$

The maximum absolute difference between each training function and its chosen ideal function is determined:

$$\text{MaxDev}_k = \max_{i=1}^{N} |y_{\text{train}, k}^{(i)} - y_{\text{ideal}, j^*}^{(i)}|$$

The mapping threshold is defined by scaling the maximum deviation:

$$\text{Threshold}_k = \text{MaxDev}_k \times \sqrt{2}$$

### Model Matching Results

```text
Training Func   Chosen Ideal   Sum Squared Dev   Max Dev   Threshold (sqrt(2))
y1              Y13            34.0807           0.4992    0.7060
y2              Y24            33.4518           0.4990    0.7057
y3              Y36            35.5727           0.4989    0.7056
y4              Y40            34.9989           0.4998    0.7068
```

### Test Data Classification Criterion
For each test coordinate (x, y), the absolute deviation Delta y to each chosen ideal function is evaluated:

$$\Delta y = |y - y_{\text{ideal}, k}(x)|$$

If Delta y does not exceed Threshold k, the point qualifies. When multiple functions qualify, the point is assigned to the function with the smallest deviation.
- Total test points evaluated: 100
- Successfully assigned points: 34
  - Function Y13: 8 points
  - Function Y24: 9 points
  - Function Y36: 10 points
  - Function Y40: 7 points
- Unassigned points: 66

## 3. Architecture and Design Principles

The application adheres to clean object-oriented design and Python best practices taught in the course:
- Object-Oriented Programming (Unit 2.2): The BaseDatasetProcessor base class encapsulates file resolution, pandas loading, and base validation. Specialized subclasses (TrainingDatasetProcessor, IdealDatasetProcessor, TestDatasetProcessor) extend functionality via constructor delegation (super().__init__()).
- Iterators and Generators (Unit 2.3): TestDatasetProcessor implements stream_test_data() using the generator protocol (yield) to stream coordinates line-by-line directly from disk without bulk memory preloading.
- Custom Exception Hierarchy (Unit 4.3): User-defined exceptions derive from AssignmentBaseException and Python standard Exception:
  - DataLoadingError: Missing or unreadable files.
  - DataValidationError: Schema, missing column, or non-numeric data failures.
  - FunctionMatchingError: Regression evaluation anomalies.
  - DatabaseOperationError: Schema compilation or insertion failures.
- Database Modeling via SQLAlchemy (Unit 3.5): DatabaseManager creates tables using SQLAlchemy Core metadata and connection pools:
  - Table 1 (training_data): X, Y1 (training func), Y2 (training func), Y3 (training func), Y4 (training func).
  - Table 2 (ideal_functions): X, Y1 (ideal func) through Y50 (ideal func).
  - Table 3 (test_data_mapping): id, X (test func), Y (test func), Delta Y (test func), No. of ideal func.
- Data Visualization via Bokeh (Unit 3.4.3): DataVisualizer builds interactive figures with Bokeh Band tolerance zones and HoverTool coordinate inspection, written to outputs/visualization.html.
- Automated Testing (Unit 5.3): test_assignment.py provides 21 automated tests deriving from unittest.TestCase.

## 4. Project Directory Structure

```text
DLMDSPWP01/
│
├── assignment/                             # Core Python package (Unit 1.5)
│   ├── __init__.py                         # Package exports
│   ├── exceptions.py                       # Custom exception hierarchy (Unit 4.3)
│   ├── processors.py                       # Base and derived processors (Unit 2.2 & 2.3)
│   ├── database.py                         # SQLAlchemy database manager (Unit 3.5)
│   └── visualizer.py                       # Bokeh visualization engine (Unit 3.4.3)
│
├── notebooks/                              # Jupyter analysis notebook (Unit 5.1)
│   └── prototype_analysis.ipynb            # End-to-end prototyping workflow
│
├── tests/                                  # Unit testing suite (Unit 5.3)
│   ├── __init__.py                         # Test package marker
│   └── test_assignment.py                  # Automated unittest cases
│
├── data/                                   # Input datasets (Local / Untracked)
│   ├── train.csv                           # Training functions
│   ├── ideal.csv                           # 50 ideal functions
│   └── test.csv                            # Test points
│
├── outputs/                                # Generated artifacts (Local / Untracked)
│   ├── assignment.db                       # Compiled SQLite database
│   ├── visualization.html                  # Interactive Bokeh dashboard
│   └── figures/                            # High-resolution publication PNGs
│
├── main.py                                 # Client entry point
├── requirements.txt                        # Pinned dependencies (Unit 5.2)
└── README.md                               # Project documentation
```

## 5. Installation and Setup

1. Clone or download the repository to your local machine.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 6. Execution Instructions

### Running the End-to-End Pipeline
Execute the main application client:
```bash
python main.py
```
This runs the full workflow:
1. Loads and validates all datasets.
2. Compiles SQLite schema in outputs/assignment.db and saves Tables 1 and 2.
3. Selects ideal functions Y13, Y24, Y36, Y40 using Least-Squares.
4. Streams test data line-by-line, maps qualifying points, and saves Table 3.
5. Generates the interactive dashboard at outputs/visualization.html.

### Running the Automated Test Suite
Run the test runner across the test package:
```bash
python -m unittest discover tests
```
All 13 unit tests execute and validate custom exceptions, loaders, mathematical matching, line-by-line generator streaming, and SQLite persistence.

### Running the Prototyping Notebook
Launch the Jupyter Notebook:
```bash
jupyter notebook notebooks/prototype_analysis.ipynb
```
The notebook executes the exploratory workflow across 8 sequential steps.

## 7. Version Control Workflow (Additional Task 1.3)

Based on Unit 6 of the course book, the Git workflow for collaborative development on the team develop branch is:

1. Clone the develop branch locally:
   ```bash
   git clone -b develop https://github.com/ainaomotayo/DLMDSPWP01.git
   cd DLMDSPWP01
   ```
2. Create and checkout a dedicated feature branch:
   ```bash
   git checkout -b feature/model-matching
   ```
3. Stage modified and newly created files:
   ```bash
   git add assignment/ main.py tests/ requirements.txt notebooks/ README.md
   ```
4. Commit the changes:
   ```bash
   git commit -m "Implement ideal function matching pipeline and unit tests"
   ```
5. Push the feature branch to the remote repository:
   ```bash
   git push origin feature/model-matching
   ```
6. Open a Pull Request on GitHub targeting develop branch for peer review and merge.
