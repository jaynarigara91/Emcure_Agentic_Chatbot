import mysql.connector
from src.logger import logging
from mysql.connector import Error
import yaml

with open('src\config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

class SQLChatHistory:
    def __init__(self):
        self.user_id = config['config']['user_id']
        self.connection = self.init_db()  # Initialize the connection here

    def init_db(self):
    
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='123',
                database='chat_history'
            )
            cursor = connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(255),
                    role VARCHAR(50),
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            connection.commit()
            logging.info('MySql Database Connection Successfull...')
            return connection  # Return the connection object
            
        except Error as e:
            print(f"Error initializing database: {e}")
            return None

    def save_message(self, role: str, content: str):
    
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO chat_history (user_id, role, content)
                VALUES (%s, %s, %s)
            ''', (self.user_id, role, content))
            self.connection.commit()
        except Error as e:
            print(f"Error saving message: {e}")
        finally:
            if cursor:
                cursor.close()

    def get_chat_history(self):

        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT role, content
                FROM chat_history
                WHERE user_id = %s
                ORDER BY timestamp ASC
            ''', (self.user_id,))
            history = cursor.fetchall()
           
            history_str = "\n".join([f"{role}: {content}" for role, content in history])
            return history_str
        except Error as e:
            print(f"Error retrieving chat history: {e}")
            return ""
        finally:
            if cursor:
                cursor.close()

    def close_connection(self):
        """Close the MySQL connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()