from dotenv import load_dotenv
try:
    from hdbcli import dbapi
except ImportError:  # hdbcli not needed when running against the local mock DB
    dbapi = None
import ast, traceback, os, re, json
from datetime import date
from langchain_community.utilities import SQLDatabase
from urllib.parse import quote_plus

# ── Dummy connection settings (override via .env) ───────────────────────────
DEFAULT_SQL_ENDPOINT = "hana-demo.example.com"
DEFAULT_DB_USER = "DEMO_USER"
DEFAULT_DB_PASSWORD = "dummy-password"
DEFAULT_DB_SCHEMA = "WAREHOUSE_DEMO"
MOCK_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_services", "mock_warehouse.db")


def _mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "true").lower() == "true"


def _mock_db_connection():
    """SQLite file seeded from the mock OData data; stands in for the HANA schema."""
    import sqlite3
    from mock_services.mock_odata_server import DATA
    conn = sqlite3.connect(MOCK_DB_PATH)
    for table, rows in DATA.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        quoted = ", ".join(f'"{c}"' for c in cols)
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({quoted})')
        if conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0:
            marks = ", ".join("?" * len(cols))
            conn.executemany(
                f'INSERT INTO "{table}" VALUES ({marks})',
                [tuple(None if r.get(c) is None else str(r.get(c)) for c in cols) for r in rows])
    conn.commit()
    return conn


date_pattern = re.compile(r"\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)")
class Utils:
    def __init__(self):
        load_dotenv()

    def initialize_llm4(self,temperature = 0):
        if _mock_mode():
            from mock_services.mock_llm import MockChatModel
            return MockChatModel()
        from gen_ai_hub.proxy.langchain import init_llm
        model= init_llm('gpt-4o', max_tokens=16384, temperature=0, verbose=True)
        return model

    def initialize_db(self):
        if _mock_mode():
            self.db_conn = _mock_db_connection()
            return self.db_conn
        self.db_conn = dbapi.connect(address=os.environ.get("sql_endpoint", DEFAULT_SQL_ENDPOINT),
                        port=443,
                        user=os.environ.get("dbuser", DEFAULT_DB_USER),
                        password=os.environ.get("dbpassword", DEFAULT_DB_PASSWORD))
        return self.db_conn

    def initialize_db_agent(self):
        if _mock_mode():
            _mock_db_connection().close()
            return SQLDatabase.from_uri(f"sqlite:///{MOCK_DB_PATH}")
        address=os.environ.get("sql_endpoint", DEFAULT_SQL_ENDPOINT)
        port=443
        user=os.environ.get("dbuser", DEFAULT_DB_USER)
        password=os.environ.get("dbpassword", DEFAULT_DB_PASSWORD)
        encoded_password = quote_plus(password)
        connection_uri = f"hana+hdbcli://{user}:{encoded_password}@{address}:{port}"
        # Initialize the SQLDatabase object
        db = SQLDatabase.from_uri(connection_uri,
                                include_tables=None,  # Include all tables
                                schema=os.environ.get("db_schema", DEFAULT_DB_SCHEMA))
        return db

    def initialize_embedding(self):
        """
        Initializes the embedding model using the OpenAIEmbeddings class with a
        specified proxy model name.

        Returns:
            OpenAIEmbeddings: The initialized embeddings object.
        """

        from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002')
        return self.embeddings

    def db_run_query(self,sql_query, params = None):
        self.initialize_db()
        conn = self.db_conn
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(sql_query, params)
                conn.commit()
                return None
            else:
                res_str = self.db_run(sql_query)
                if(res_str):
                    # res_str = str(res_str).replace('ItemText','Defect Identified')
                    # res_str = str(res_str).replace('TaskText','Resolution')
                    res_str = str(res_str).replace("datetime.datetime","")
                    res_str = str(res_str).replace("datetime.date","")
                    res_str = str(res_str).replace("Decimal","")
                    res_str = re.sub(date_pattern, self.replace_date, str(res_str))
                    result = ast.literal_eval(res_str)
                else:
                    print("no data available")
                    result = []
                return result
        except Exception as e:
            print(f"Database query error: {e}")
            # Assuming conn is available in the except block
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def replace_date(self,match):
        # Function to replace matched dates with "YYYY-MM-DD" format
        try:
            year, month, day = map(int, match.groups())
            date_obj = date(year, month, day)
            return f'"{date_obj.strftime("%Y-%m-%d")}"'
        except ValueError:
            return match.group(0)

    def db_run(self,sql):
        print("Running db query")
        self.initialize_db()
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql)
            if cursor.description is None:
                # No result set available
                self.db_conn.commit()  # Important for INSERT/UPDATE
                return []  # or return [{"message": "DML executed successfully"}]
            columns = cursor.description
            # result = cursor.fetchall()
            # print(result)
            result = [{columns[index][0]:column for index, column in enumerate(value)} for value in cursor.fetchmany()]
            # print(result)
            # result = cursor.fetchall()
            return result
        except Exception as e:
            print(traceback.format_exc())
        finally:
            cursor.close()

    def correct_json(self,json_str):
        """
        Correct curly braces in JSON string
        Args:
            json string (String): Json string response generated by LLM
        Returns:
            correct json string
        """
        # logger.info("correcting json")
        try:
            # Clean json string
            pattern = r'^```json\s*(.*?)\s*```$'
            cleaned_string = re.sub(pattern, r'\1', json_str, flags=re.DOTALL)
            # Add missing closing curly braces
            num_opening_braces = cleaned_string.count('{')
            num_closing_braces = cleaned_string.count('}')
            if num_opening_braces > num_closing_braces:
                cleaned_string += '}' * (num_opening_braces - num_closing_braces)
            # Remove extra closing curly braces
            while cleaned_string.count('}') > cleaned_string.count('{'):
                cleaned_string = cleaned_string[:-1]

            # Add missing closing square braces
            num_opening_braces = cleaned_string.count('[')
            num_closing_braces = cleaned_string.count(']')
            if num_opening_braces > num_closing_braces:
                cleaned_string += ']' * (num_opening_braces - num_closing_braces)
            # Remove extra closing square braces
            while cleaned_string.count(']') > cleaned_string.count('['):
                cleaned_string = cleaned_string[:-1]

            return cleaned_string
        except Exception as e:
            return str(e)

    def get_all_table_columns_by_type(self, schema_name, table_names):
        conn = self.initialize_db()
        cursor = conn.cursor()

        table_column_info = {}

        for table in table_names:
            column_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{schema_name}'
                AND TABLE_NAME = '{table}'
                ORDER BY POSITION
            """
            print(column_query)
            cursor.execute(column_query)
            columns = cursor.fetchall()

            # Build CREATE VIEW-like string
            header = f"\"{schema_name}\".\"{table}\" (\n"
            columns_strs = [
                f"    {col[0]} {col[1]}({col[2]})" for col in columns
            ]
            header += ", \n".join(columns_strs) + "\n)"

            # Step 2: Fetch 3 sample rows
            data_query = f'SELECT * FROM "{schema_name}"."{table}" LIMIT 3'
            cursor.execute(data_query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]

            # Format rows like a table with column headers
            sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{table}\":\n"

            # Tab-separated header
            sample_block += "\t".join(col_names) + "\n"

            # Tab-separated values
            for row in rows:
                row_str = "\t".join(str(val) if val is not None else "None" for val in row)
                sample_block += row_str + "\n"

            sample_block += "*/"

            # Combine all
            table_column_info[table] = header + sample_block

        cursor.close()
        conn.close()
        return table_column_info

    def get_all_table_columns_by_type_with_placeholders(self, schema_name, table_names, placeholders):
        conn = self.initialize_db()
        cursor = conn.cursor()

        table_column_info = {}

        for table in table_names:
            column_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{schema_name}'
                AND TABLE_NAME = '{table}'
                ORDER BY POSITION
            """
            print(column_query)
            cursor.execute(column_query)
            columns = cursor.fetchall()

            # Build CREATE VIEW-like string
            header = f"\"{schema_name}\".\"{table}\" (\n"
            columns_strs = [
                f"    {col[0]} {col[1]}({col[2]})" for col in columns
            ]
            header += ", \n".join(columns_strs) + "\n)"

            # Step 2: Fetch 3 sample rows
            data_query = f'SELECT * FROM "{schema_name}"."{table}"({placeholders}) LIMIT 3'
            cursor.execute(data_query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]

            # Format rows like a table with column headers
            sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{table}\":\n"

            # Tab-separated header
            sample_block += "\t".join(col_names) + "\n"

            # Tab-separated values
            for row in rows:
                row_str = "\t".join(str(val) if val is not None else "None" for val in row)
                sample_block += row_str + "\n"

            sample_block += "*/"

            # Combine all
            table_column_info[table] = header + sample_block

        cursor.close()
        conn.close()
        return table_column_info

    def get_all_table_columns_by_type_with_placeholders_cv(self, schema_name, table_names, placeholders):
        conn = self.initialize_db()
        cursor = conn.cursor()

        table_column_info = {}

        for table in table_names:
            # Step 1: Get column metadata WITHOUT parameters (clean table name)
            # Extract just the table name without any parameter syntax
            # If the table name has parameters attached, strip them
            clean_table_name = table
            if '(' in table:
                clean_table_name = table.split('(')[0].strip()

            column_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{schema_name}'
                AND TABLE_NAME = '{clean_table_name}'
                ORDER BY POSITION
            """
            print(f"Column query: {column_query}")
            cursor.execute(column_query)
            columns = cursor.fetchall()

            # Build CREATE VIEW-like string
            header = f"\"{schema_name}\".\"{clean_table_name}\" (\n"
            columns_strs = []
            for col in columns:
                if col[2]:  # If LENGTH exists
                    columns_strs.append(f"    {col[0]} {col[1]}({col[2]})")
                else:
                    columns_strs.append(f"    {col[0]} {col[1]}")
            header += ", \n".join(columns_strs) + "\n)"

            # Step 2: Fetch 3 sample rows WITH parameters
            try:
                # For calculation views with placeholders, use the placeholder syntax
                data_query = f'SELECT * FROM "{schema_name}"."{clean_table_name}"({placeholders}) LIMIT 3'
                print(f"Data query: {data_query}")
                cursor.execute(data_query)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]

                # Format rows like a table with column headers
                sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{clean_table_name}\":\n"

                # Tab-separated header
                sample_block += "\t".join(col_names) + "\n"

                # Tab-separated values
                for row in rows:
                    row_str = "\t".join(str(val) if val is not None else "None" for val in row)
                    sample_block += row_str + "\n"

                sample_block += "*/"
            except Exception as e:
                # If parameterized query fails, try without parameters
                print(f"Parameterized query failed: {e}")
                try:
                    data_query = f'SELECT * FROM "{schema_name}"."{clean_table_name}" LIMIT 3'
                    cursor.execute(data_query)
                    rows = cursor.fetchall()
                    col_names = [desc[0] for desc in cursor.description]

                    sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{clean_table_name}\" (no parameters):\n"
                    sample_block += "\t".join(col_names) + "\n"
                    for row in rows:
                        row_str = "\t".join(str(val) if val is not None else "None" for val in row)
                        sample_block += row_str + "\n"
                    sample_block += "*/"
                except Exception as e2:
                    sample_block = f"\n/*\nError fetching sample data: {e2}\n*/"

            # Combine all
            table_column_info[table] = header + sample_block

        cursor.close()
        conn.close()
        return table_column_info

    def get_all_table_columns_by_type_with_multiple_placeholders_cv(self, schema_name, table_names, placeholders):
        conn = self.initialize_db()
        cursor = conn.cursor()

        table_column_info = {}

        for table in table_names:
            # Step 1: Get column metadata WITHOUT parameters (clean table name)
            clean_table_name = table
            if '(' in table:
                clean_table_name = table.split('(')[0].strip()

            column_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{schema_name}'
                AND TABLE_NAME = '{clean_table_name}'
                ORDER BY POSITION
            """
            print(f"Column query: {column_query}")
            cursor.execute(column_query)
            columns = cursor.fetchall()

            # Build CREATE VIEW-like string
            header = f"\"{schema_name}\".\"{clean_table_name}\" (\n"
            columns_strs = []
            for col in columns:
                if col[2]:  # If LENGTH exists
                    columns_strs.append(f"    {col[0]} {col[1]}({col[2]})")
                else:
                    columns_strs.append(f"    {col[0]} {col[1]}")
            header += ", \n".join(columns_strs) + "\n)"

            # Step 2: Fetch 3 sample rows WITH parameters
            try:
                # Build the complete query with placeholders
                if placeholders and placeholders.strip():
                    data_query = f'SELECT TOP 3 * FROM "{schema_name}"."{clean_table_name}"({placeholders})'
                else:
                    data_query = f'SELECT TOP 3 * FROM "{schema_name}"."{clean_table_name}"'

                print(f"Data query: {data_query}")
                cursor.execute(data_query)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]

                # Format rows like a table with column headers
                sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{clean_table_name}\":\n"
                if placeholders and placeholders.strip():
                    sample_block += f"With parameters: {placeholders}\n"

                # Tab-separated header
                sample_block += "\t".join(col_names) + "\n"

                # Tab-separated values
                for row in rows:
                    row_str = "\t".join(str(val) if val is not None else "None" for val in row)
                    sample_block += row_str + "\n"

                sample_block += "*/"
            except Exception as e:
                # If parameterized query fails, try without parameters
                print(f"Parameterized query failed: {e}")
                try:
                    data_query = f'SELECT TOP 3 * FROM "{schema_name}"."{clean_table_name}"'
                    cursor.execute(data_query)
                    rows = cursor.fetchall()
                    col_names = [desc[0] for desc in cursor.description]

                    sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{clean_table_name}\" (no parameters):\n"
                    sample_block += "\t".join(col_names) + "\n"
                    for row in rows:
                        row_str = "\t".join(str(val) if val is not None else "None" for val in row)
                        sample_block += row_str + "\n"
                    sample_block += "*/"
                except Exception as e2:
                    sample_block = f"\n/*\nError fetching sample data: {e2}\n*/"

            # Combine all
            table_column_info[table] = header + sample_block

        cursor.close()
        conn.close()
        return table_column_info

    def get_table_info_and_sample_formatted(self, view_name):
        conn = self.initialize_db()
        cursor = conn.cursor()
        schema_name = DEFAULT_DB_SCHEMA
        # Step 1: Fetch column metadata
        metadata_query = f"""
        SELECT
                COLUMN_NAME,
                DATA_TYPE_NAME,
                LENGTH
            FROM SYS.VIEW_COLUMNS
            WHERE SCHEMA_NAME = '{schema_name.upper()}'
            AND VIEW_NAME   = '{view_name}'
            ORDER BY POSITION
        """

        cursor.execute(metadata_query)
        column_info = cursor.fetchall()

        # Build CREATE VIEW-like string
        header = f"\"{schema_name}\".\"{view_name}\" (\n"
        columns_strs = [
            f"    {col[0]} {col[1]}({col[2]})" for col in column_info
        ]
        header += ", \n".join(columns_strs) + "\n)"

        # Step 2: Fetch 3 sample rows
        data_query = f'SELECT * FROM "{schema_name}"."{view_name}" LIMIT 3'
        cursor.execute(data_query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]

        # Format rows like a table with column headers
        sample_block = f"\n/*\n3 rows from \"{schema_name}\".\"{view_name}\":\n"

        # Tab-separated header
        sample_block += "\t".join(col_names) + "\n"

        # Tab-separated values
        for row in rows:
            row_str = "\t".join(str(val) if val is not None else "None" for val in row)
            sample_block += row_str + "\n"

        sample_block += "*/"

        cursor.close()
        conn.close()

        # Combine all
        return header + sample_block

    def data_transform(self, data_list, cv, sub_agent):
        if not data_list:
            return None

        # Load col_map
        col_map_path = os.path.join(os.path.dirname(__file__), 'col_map.json')
        with open(col_map_path, 'r') as f:
            col_map = json.load(f)

        # Get the mapping for cv
        mapping = col_map.get(cv, {})

        # Extract headers from first dictionary keys
        headers = list(data_list[0].keys())

        # Replace with friendly names
        friendly_headers = [mapping.get(h, h) for h in headers]

        # Extract rows (just the values)
        rows = []
        for row in data_list:
            rows.append(list(row.values()))

        # Create the final structure
        sub_agent_data = {
            sub_agent: {
                # "headers": friendly_headers,
                # Backwards-compatible key name used by agents: 'columns'
                "columns": friendly_headers,
                "rows": rows
            }
        }

        return sub_agent_data