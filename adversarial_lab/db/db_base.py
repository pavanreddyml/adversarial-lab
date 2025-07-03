from abc import ABC, abstractmethod

from typing import Optional, List

class DB(ABC):
    """
    Abstract base class for database interactions.

    This class defines the core methods required for a database implementation, including 
    connection validation, table management, data insertion, and connection closure.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the database object.
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> None:
        """
        Validate the connection to the database. Checks whether the database connection is active and functional.

        Raises:
            ConnectionError: If the connection is not valid or cannot be established.
        """
        pass

    @abstractmethod
    def create_table(self, table_name: str, schema: dict, force: bool = False, primary_key_col_name: str = "id") -> None:
        """
        Create a table in the database.

        Parameters:
            table_name (str): The name of the table to create.
            schema (dict): A dictionary defining the table schema, where keys are column names 
                and values are data types.
            force (bool, optional): If True, deletes the table if it already exists before creating 
                a new one. Defaults to False.

        Raises:
            ValueError: If the table already exists and `force` is set to False.

        Notes:
            - If `force` is True and the table exists, it will be dropped before creation.
            - Schema must be defined according to the database's expected format.
        """
        pass

    @abstractmethod
    def delete_table(self, table_name: str) -> None:
        """
        Delete a table from the database.

        Parameters:
            table_name (str): The name of the table to be deleted.

        Notes:
            - This operation is irreversible; all data in the table will be lost.
        """
        pass
    
    @abstractmethod
    def insert(self, table_name: str, data: dict) -> None:
        """
        Insert data into the specified table.

        Parameters:
            table_name (str): The name of the table where the data should be inserted.
            data (dict): A dictionary containing column-value pairs representing the record.

        Notes:
            - The keys in `data` must match column names in the database schema.
            - If the table does not exist, this operation should raise an error in concrete implementations.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Terminate any open database connections.

        Notes:
            - After calling `close()`, further database operations should not be attempted.
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[List] = None) -> None:
        """
        Execute a raw SQL query.

        Parameters:
            query (str): The SQL query to be executed.

        Notes:
            - This method should handle the execution of any SQL command, including SELECT, 
              INSERT, UPDATE, DELETE, etc.
            - Ensure that the query is safe from SQL injection attacks.
        """
        pass
