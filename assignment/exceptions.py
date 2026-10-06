'''
Module exceptions
Defines custom exception classes for data processing and database operations.
Follows Unit 4.3 of the course book.
'''


class AssignmentBaseException(Exception):
    '''
    Base exception for assignment processing errors.
    Inherits from the Python standard Exception class.
    '''
    def __init__(self, message=None):
        super().__init__(message)
        self.message = message


class DataLoadingError(AssignmentBaseException):
    '''
    Raised when a dataset file cannot be loaded or found.
    '''
    def __init__(self, filepath, message=None):
        if message is None:
            message = f"Failed to load dataset file: {filepath}"
        super().__init__(message)
        self.filepath = filepath


class DataValidationError(AssignmentBaseException):
    '''
    Raised when dataset contents do not match expected structure or types.
    '''
    def __init__(self, dataset_name, details):
        message = f"Validation failure in {dataset_name}: {details}"
        super().__init__(message)
        self.dataset_name = dataset_name
        self.details = details


class FunctionMatchingError(AssignmentBaseException):
    '''
    Raised when ideal function selection or regression matching fails.
    '''
    def __init__(self, message=None):
        if message is None:
            message = "Error during ideal function regression matching."
        super().__init__(message)


class DatabaseOperationError(AssignmentBaseException):
    '''
    Raised when SQLite database schema compilation or insertion fails.
    '''
    def __init__(self, operation, details):
        message = f"Database operation '{operation}' failed: {details}"
        super().__init__(message)
        self.operation = operation
        self.details = details
