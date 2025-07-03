import sqlite3
import json
import os

class SqlliteDB:
    db_types = {
        'int': 'INTEGER',
        'str': 'TEXT',
        'float': 'REAL',
        'bool': 'INTEGER',
        'json': 'TEXT',
        'blob': 'BLOB'
    }

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            print(f"Database {self.db_path} does not exist. Creating new database.")
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def validate_connection(self) -> bool:
        return True

    def create_table(self, 
                     table_name: str, 
                     schema: dict, 
                     force: bool = False,
                     primary_key_col_name: str = "id"
                     ) -> None:
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
        )
        table_exists = self.cursor.fetchone()

        if table_exists:
            if force:
                print(f"Warning: Table '{table_name}' already exists. Recreating the table.")
                self.delete_table(table_name)
            else:
                raise ValueError(f"Table '{table_name}' already exists. Use force=True to delete and recreate it.")
        

        columns = [f"{primary_key_col_name} INTEGER PRIMARY KEY AUTOINCREMENT"]
        for column_name, column_type in schema.items():
            if column_name == primary_key_col_name:
                raise ValueError(f"Column name '{primary_key_col_name}' is reserved for primary key. Please choose a different name for columns.")
            if column_type not in self.db_types:
                raise ValueError(f"Unsupported column type: {column_type}")
            columns.append(f"{column_name} {self.db_types[column_type]}")
        columns_str = ", ".join(columns)
        query = f"CREATE TABLE {table_name} ({columns_str});"
        self.cursor.execute(query)
        self.connection.commit()

    def delete_table(self, 
                     table_name: str
                     )-> None:
        query = f"DROP TABLE IF EXISTS {table_name};"
        self.cursor.execute(query)
        self.connection.commit()

    def insert(self, 
               table_name: str, 
               data: dict
               ) -> None:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        values = []
        for value in data.values():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            values.append(value)
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"
        self.cursor.execute(query, values)
        self.connection.commit()

    def execute_query(self, 
                     query: str, 
                     params: tuple = ()
                     ) -> list | None:
        self.cursor.execute(query, params)
        if self.cursor.description is None:
            self.connection.commit()
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def close(self) -> None:
        self.connection.close()
