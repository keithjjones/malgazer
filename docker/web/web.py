from flask import Flask, render_template, abort, redirect, url_for, flash, request
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from wtforms import RadioField, StringField, PasswordField
from wtforms.validators import DataRequired, Email
from flask_wtf.file import FileField, FileRequired
from flask_wtf.csrf import CSRFProtect, CSRFError
import requests
import os
import json
import datetime
import sys
import logging
import logging.handlers
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))
from library.files import Sample
from docker.db_models.models import Submission, WebRequest, setup_database, db


# Global values
API_URL = "http://api:8888"
POSSIBLE_CLASSIFICATIONS = [
    ('Trojan', 'Trojan'),
    ('Virus', 'Virus'),
    ('Worm', 'Worm'),
    ('Backdoor', 'Backdoor'),
    ('Ransomware', 'Ransomware'),
    ('PUP', 'PUP'),
    ('Unknown', 'Unknown')
]
MULTIUSER = bool(int(os.environ['MULTIUSER']))

# Initialize and configure the Flask website.
app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="secretkey 1",
    WTF_CSRF_SECRET_KEY="secretkey 2"
))
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:malgazer@db/postgres'
db.init_app(app)
csrf = CSRFProtect(app)
applogger = app.logger
file_handler = logging.handlers.TimedRotatingFileHandler("/logs/web.log", when='midnight', backupCount=30)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
applogger.setLevel(logging.DEBUG)
applogger.addHandler(file_handler)


class SubmissionForm(FlaskForm):
    """
    The submission form.
    """
    sample = FileField(validators=[FileRequired()])
    classification = RadioField('name', choices=POSSIBLE_CLASSIFICATIONS, default='Unknown', validators=[DataRequired()])


class LoginForm(FlaskForm):
    """
    The login form.
    """
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])


@app.route('/')
def main():
    """
    The main page.
    """
    return render_template('main.html')


@app.route('/login')
def login():
    """
    The login page.
    """
    ip_addr = request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])
    form = LoginForm()
    if form.validate_on_submit():
        pass
    return render_template('login.html', form=form)


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/submit', methods=('GET', 'POST'))
def submit():
    """
    The sample submission page.
    """
    ip_addr = request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])
    form = SubmissionForm()
    if form.validate_on_submit():
        f = form.sample.data
        s = Sample(frommemory=f.stream.read())
        files = {'file': s.rawdata}
        data = {'classification': form.classification.data}
        url = "{0}/submit".format(API_URL)
        try:
            req = requests.post(url, files=files, data=data)
        except Exception as exc:
            flash('Exception while sending file to API!')
            app.logger.exception('Error submitting sample: {0} - Exception: {1}'.format(s.sha256, exc))
            return redirect(url_for('submit'))
        if req.status_code != 200:
            flash("API FAILURE - HTTP Code: {0}".format(req.status_code))
            app.logger.error('Submit API did not return 200: {0}'.format(req))
            return redirect(url_for('submit'))
        submit_time = datetime.datetime.now()
        req = WebRequest(sha256=s.sha256, time=submit_time,
                         possible_classification=form.classification.data,
                         ip_address=ip_addr)
        db.session.add(req)
        db.session.commit()
        app.logger.info('Submitted sample: {0} from IP: {1}'.format(s.sha256, ip_addr))
        return redirect(url_for('history'))
    return render_template('submit.html', form=form)


@app.route('/history')
def history():
    """
    The submission history page.
    """
    url = "{0}/history".format(API_URL)
    try:
        req = requests.get(url)
    except Exception as exc:
        flash('Exception while pulling history from API!')
        app.logger.exception('Exception while pulling history - Exception: {0}'.format(exc))
        return redirect(url_for('main'))
    if req.status_code != 200:
        flash("API FAILURE - HTTP Code: {0}".format(req.status_code))
        app.logger.error('History API did not return 200: {0}'.format(req))
        return redirect(url_for('main'))
    history = json.loads(req.text)
    return render_template('history.html', history=history)


@app.route('/api')
def api():
    """
    The API information page.
    """
    return render_template('api.html')


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
