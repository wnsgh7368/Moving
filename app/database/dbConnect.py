import mysql.connector
from mysql.connector import pooling

class DbConnect:
    def __init__(self, host, user, password, database):
        self.db_config = {
            "pool_name": "mypool",
            "pool_size": 5,
            "host": host,
            "user": user,
            "password": password,
            "database": database
        }
        self.connection_pool = self.create_connection_pool()
        self.create_table()

    def create_connection_pool(self):
        try:
            pool = mysql.connector.pooling.MySQLConnectionPool(**self.db_config)
            return pool
        except mysql.connector.Error as e:
            print(f"풀 생성 오류: {e}")
            raise

    def get_connection(self):
        return self.connection_pool.get_connection()

    def create_table(self):
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_count (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                count INT
            )
            """)
            connection.commit()
        except mysql.connector.Error as e:
            print(f"테이블 생성 오류: {e}")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def insert_count(self, count):
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            insert_query = """
            INSERT INTO person_count (timestamp, count) 
            VALUES (NOW(), %s)
            """
            cursor.execute(insert_query, (count,))
            connection.commit()
            
        except mysql.connector.Error as e:
            print(f"데이터 삽입 오류: {e}")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def get_latest_count(self):
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
            SELECT count 
            FROM person_count 
            WHERE count IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] if result else 0
            
        except mysql.connector.Error as e:
            print(f"데이터 조회 오류: {e}")
            return 0
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def close(self):
        """Close the database connection pool."""
        try:
            if self.connection_pool:
                self.connection_pool.close()
                print("Database connection pool closed.")
        except Exception as e:
            print(f"Error closing database connection pool: {e}")