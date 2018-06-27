from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
import json
import os
from werkzeug.utils import secure_filename
from ...library.files import Sample
import datetime
import sqlalchemy
from sqlalchemy.dialects import postgresql
import glob

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:malgazer@db/postgres'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
db = SQLAlchemy(app)

SAMPLES_DIRECTORY = "/samples"


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sha256 = db.Column(db.String(80), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    classification = db.Column(db.String(120), nullable=True)
    ip_address = db.Column(postgresql.INET)


db.create_all()


@app.route('/initdb')
def initdb():
    engine = sqlalchemy.create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    Submission.__table__.drop(engine)
    db.create_all()
    for f in glob.glob(os.path.join(SAMPLES_DIRECTORY, '*')):
        os.remove(f)
    return json.dumps({})


@app.route('/')
def main():
    return json.dumps({})


@app.route('/submit', methods=('POST',))
def submit():
    if 'ip_address' in request.form:
        ip_addr = request.form.get('ip_address', 'None')
    else:
        ip_addr = request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])
    if 'file' not in request.files:
        return "ERROR"
    file = request.files['file']
    f = file.stream.read()
    s = Sample(frommemory=f)
    filename = secure_filename(s.sha256)
    filepath = os.path.join(SAMPLES_DIRECTORY, filename)
    if not os.path.isfile(filepath):
        with open(filepath, 'wb') as f_out:
            f_out.write(s.rawdata)
    submit_time = datetime.datetime.now()
    submission = Submission(sha256=s.sha256, time=submit_time, ip_address=ip_addr)
    return_data = {'id': submission.id, 'sha256': submission.sha256, 'time': str(submission.time), 'ip_address': submission.ip_address}
    db.session.add(submission)
    db.session.commit()
    return json.dumps(return_data), 200


@app.route("/history")
def history():
    submissions = Submission.query.order_by(Submission.id.desc()).all()
    return_data = []
    for s in submissions:
        return_data.append({'id': s.id,
                            'sha256': s.sha256, 'time': str(s.time),
                            'classification': s.classification,
                            'ip_address': s.ip_address})
    return json.dumps(return_data)


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8888)