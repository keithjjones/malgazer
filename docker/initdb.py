from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
import os
import sys
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))
from malgazer.docker.api.malgazer import Submission
from malgazer.docker.web.malgazer import WebRequest

# Initialize and configure the Flask API
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:malgazer@db/postgres'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
db = SQLAlchemy(app)

with app.test_request_context():
    db.drop_all()
    db.create_all()