from flask import Flask

app = Flask(__name__)

from app import routes  # Importing the routes (view functions)
