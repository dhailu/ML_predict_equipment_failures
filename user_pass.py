import os
import sys
from dotenv import load_dotenv
from src.exception import CustomeException

# Load environment variables from .env file
load_dotenv()

def get_pass():
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        if not user or not password:
            raise ValueError("Missing DB_USER or DB_PASS in .env or environment")
        return user, password
    except Exception as e:
        raise CustomeException(e, sys)
