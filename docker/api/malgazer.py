from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
import json
import os
from werkzeug.utils import secure_filename
from ...library.files import Sample
import datetime
import sqlalchemy
import glob

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:malgazer@db/postgres'
db = SQLAlchemy(app)

SAMPLES_DIRECTORY = "/samples"


class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sha256 = db.Column(db.String(80), nullable=False)
    time = db.Column(db.DateTime, nullable=False)


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
    return_data = {'sha256': s.sha256}
    submission = Submission(sha256=s.sha256, time=datetime.datetime.now())
    db.session.add(submission)
    db.session.commit()
    submissions = Submission.query.all()
    return_data = []
    for s in submissions:
        return_data.append({'sha256': s.sha256, 'time': str(s.time)})
    return json.dumps(return_data)


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)