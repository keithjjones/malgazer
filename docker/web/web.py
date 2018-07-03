from flask import Flask, render_template, abort, redirect, url_for, flash, request
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from wtforms import RadioField, StringField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired
from flask_wtf.csrf import CSRFProtect, CSRFError
import requests
import os
import json
import datetime
import sys
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))
sys.path.append(os.path.join('..', '..', '..'))
from malgazer.library.files import Sample
from malgazer.docker.db_models.models import Submission, WebRequest, setup_database, db


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


class SubmissionForm(FlaskForm):
    """
    The submission form.
    """
    sample = FileField(validators=[FileRequired()])
    classification = RadioField('name', choices=POSSIBLE_CLASSIFICATIONS, default='Unknown', validators=[DataRequired()])


@app.route('/')
def main():
    """
    The main page.
    """
    return render_template('main.html')


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
        except:
            flash('Exception while sending file to API!')
            return redirect(url_for('submit'))
        if req.status_code != 200:
            flash("API FAILURE - HTTP Code: {0}".format(req.status_code))
            return redirect(url_for('submit'))
        submit_time = datetime.datetime.now()
        req = WebRequest(sha256=s.sha256, time=submit_time,
                         possible_classification=form.classification.data,
                         ip_address=ip_addr)
        db.session.add(req)
        db.session.commit()
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
    except:
        flash('Exception while pulling history from API!')
        return redirect(url_for('main'))
    if req.status_code != 200:
        flash("API FAILURE - HTTP Code: {0}".format(req.status_code))
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