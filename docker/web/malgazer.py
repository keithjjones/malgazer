from flask import Flask, render_template, abort, redirect, url_for, flash, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.utils import secure_filename
from ...library.files import Sample
import requests
import os
import json

API_URL = "http://api:8888"

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="secretkey 1",
    WTF_CSRF_SECRET_KEY="secretkey 2"
))
csrf = CSRFProtect(app)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/submit', methods=('GET', 'POST'))
def submit():
    ip_addr = request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])
    form = Submission()
    if form.validate_on_submit():
        f = form.sample.data
        s = Sample(frommemory=f.stream.read())
        files = {'file': s.rawdata}
        data = {'ip_address': ip_addr}
        url = "{0}/submit".format(API_URL)
        try:
            req = requests.post(url, files=files, data=data)
        except:
            flash('Exception while sending file to API!')
            return redirect(url_for('submit'))
        if req.status_code != 200:
            flash("API FAILURE - HTTP Code: {0}".format(req.status_code))
            return redirect(url_for('submit'))
        return redirect(url_for('history'))
    return render_template('submit.html', form=form)


@app.route('/history')
def history():
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


class Submission(FlaskForm):
    sample = FileField(validators=[FileRequired()])


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)