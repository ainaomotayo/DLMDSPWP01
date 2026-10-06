'''
Module database
Implements SQLite database creation and CRUD operations using SQLAlchemy Core.
Follows Unit 3.5 of the course book.
'''

import os
import sqlalchemy as db
from assignment.exceptions import DatabaseOperationError


def resolve_output_path(filepath):
    '''
    Resolves output file path relative to project root if needed.
    '''
    if not filepath:
        return "outputs/assignment.db"
    if os.path.isabs(filepath):
        return filepath
    if os.path.exists(os.path.dirname(filepath)) and os.path.dirname(filepath):
        return filepath

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, filepath)


class DatabaseManager(object):
    '''
    Manages SQLite database creation and CRUD operations using SQLAlchemy.
    Conforms to the table schemas specified in the assignment document.
    '''

    def __init__(self, db_filepath="outputs/assignment.db"):
        '''
        Constructor for DatabaseManager.
        db_filepath: filename for local SQLite database.
        '''
        self.db_filepath = resolve_output_path(db_filepath)
        parent_dir = os.path.dirname(self.db_filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.db_url = f"sqlite:///{self.db_filepath}"
        self.engine = db.create_engine(self.db_url)
        self.metadata = db.MetaData()
        self.tables = {}

    def compile_database_schema(self):
        '''
        Builds Table 1, Table 2, and Table 3 schemas in SQLite database.
        '''
        try:
            # Table 1: Training Data (5 columns)
            self.tables["training_data"] = db.Table(
                "training_data",
                self.metadata,
                db.Column("X", db.Float, primary_key=True),
                db.Column("Y1 (training func)", db.Float, nullable=False),
                db.Column("Y2 (training func)", db.Float, nullable=False),
                db.Column("Y3 (training func)", db.Float, nullable=False),
                db.Column("Y4 (training func)", db.Float, nullable=False),
            )

            # Table 2: Ideal Functions (51 columns)
            ideal_cols = [db.Column("X", db.Float, primary_key=True)]
            for i in range(1, 51):
                ideal_cols.append(db.Column(f"Y{i} (ideal func)", db.Float, nullable=False))

            self.tables["ideal_functions"] = db.Table(
                "ideal_functions",
                self.metadata,
                *ideal_cols
            )

            # Table 3: Test Data Mapping (4 columns)
            self.tables["test_data_mapping"] = db.Table(
                "test_data_mapping",
                self.metadata,
                db.Column("id", db.Integer, primary_key=True, autoincrement=True),
                db.Column("X (test func)", db.Float, nullable=False),
                db.Column("Y (test func)", db.Float, nullable=False),
                db.Column("Delta Y (test func)", db.Float, nullable=False),
                db.Column("No. of ideal func", db.String(20), nullable=False),
            )

            self.metadata.create_all(self.engine)
        except Exception as error:
            raise DatabaseOperationError("Schema Creation", str(error))

    def save_training_data(self, train_df):
        '''
        Inserts training data into Table 1.
        train_df: DataFrame containing columns x, y1, y2, y3, y4.
        '''
        try:
            records = []
            for _, row in train_df.iterrows():
                records.append({
                    "X": float(row["x"]),
                    "Y1 (training func)": float(row["y1"]),
                    "Y2 (training func)": float(row["y2"]),
                    "Y3 (training func)": float(row["y3"]),
                    "Y4 (training func)": float(row["y4"])
                })

            table = self.tables["training_data"]
            with self.engine.connect() as connection:
                connection.execute(db.delete(table))
                connection.execute(db.insert(table), records)
                connection.commit()
        except Exception as error:
            raise DatabaseOperationError("Save Training Data", str(error))

    def save_ideal_functions(self, ideal_df):
        '''
        Inserts ideal functions data into Table 2.
        ideal_df: DataFrame containing columns x and y1 through y50.
        '''
        try:
            records = []
            for _, row in ideal_df.iterrows():
                entry = {"X": float(row["x"])}
                for i in range(1, 51):
                    entry[f"Y{i} (ideal func)"] = float(row[f"y{i}"])
                records.append(entry)

            table = self.tables["ideal_functions"]
            with self.engine.connect() as connection:
                connection.execute(db.delete(table))
                connection.execute(db.insert(table), records)
                connection.commit()
        except Exception as error:
            raise DatabaseOperationError("Save Ideal Functions", str(error))

    def save_test_data_mapping(self, mapped_records):
        '''
        Inserts mapped test records into Table 3.
        mapped_records: list of dictionaries containing mapped test data.
        '''
        try:
            table = self.tables["test_data_mapping"]
            with self.engine.connect() as connection:
                connection.execute(db.delete(table))
                if mapped_records:
                    connection.execute(db.insert(table), mapped_records)
                connection.commit()
        except Exception as error:
            raise DatabaseOperationError("Save Test Data Mapping", str(error))

    def fetch_records_count(self, table_name):
        '''
        Returns record count for a table.
        '''
        if table_name not in self.tables:
            raise DatabaseOperationError("Count Query", f"Unknown table: {table_name}")
        table = self.tables[table_name]
        with self.engine.connect() as connection:
            result = connection.execute(db.select(db.func.count()).select_from(table)).scalar()
            return result
