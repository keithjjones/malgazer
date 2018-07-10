from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
import json
import os
from werkzeug.utils import secure_filename
import datetime
import sqlalchemy
import glob
import multiprocessing
import threading
import sys
import logging
import logging.handlers
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))
from library.files import Sample
from library.ml import ML
from docker.db_models.models import Submission, WebRequest, setup_database, clear_database, db
import library


# Initialize and configure the Flask API
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:malgazer@db/postgres'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
db.init_app(app)

applogger = app.logger
file_handler = logging.handlers.TimedRotatingFileHandler("/logs/api.log", when='midnight', backupCount=30)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
applogger.setLevel(logging.DEBUG)
applogger.addHandler(file_handler)

# Global values
SAMPLES_DIRECTORY = "/samples"
MULTIUSER = bool(int(os.environ['MULTIUSER']))

with app.app_context():
    db.create_all()


def setup_db():
    setup_database()


@app.before_first_request
def setup():
    setup_db()


@app.route('/reset')
def reset():
    """
    Put all cleaning logic here.
    """
    clear_database()
    for f in glob.glob(os.path.join(SAMPLES_DIRECTORY, '*')):
        if os.path.isfile(f):
            os.remove(f)
    return json.dumps({'status': 'reset'}), 200


def process_sample(submission_id):
    """
    Processes a sample after it has been submitted.  This runs as a thread.

    :param submission_id:  The ID of the submission in the database.
    :return: Nothing.
    """
    sys.modules['library'] = library
    submission = Submission.query.filter_by(id=submission_id).first()
    try:
        # ml = dill.load(open(os.path.join('..', '..', 'classifier', 'ml.dill'), 'rb'))
        ml = ML.load_classifier(os.path.join('..', '..', 'classifier'))
        s = Sample(fromfile=os.path.join('/samples', submission.sha256))
        y = ml.predict_sample(s)
        submission = Submission.query.filter_by(id=submission_id).first()
        submission.status = 'Done'
        submission.classification = y
        app.logger.info('Finished processing sample: {0} as: {1}'.format(s.sha256, y))
    except Exception as exc:
        submission.status = 'Error'
        app.logger.exception('Error processing sample: {0} - Exception: {1}'.format(s.sha256, exc))
    db.session.add(submission)
    db.session.commit()


@app.route('/submit', methods=('POST',))
def submit():
    """
    Submits a sample and executes thread to process it.
    """
    ip_addr = request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])
    possible_classification = request.form.get('classification', 'Unknown')
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
    submission = Submission(sha256=s.sha256, time=submit_time, ip_address=ip_addr,
                            classification='', possible_classification=possible_classification,
                            status='Processing')
    return_data = {'id': submission.id, 'sha256': submission.sha256,
                   'time': str(submission.time), 'ip_address': submission.ip_address,
                   'classification': submission.classification,
                   'possible_classification': submission.possible_classification,
                   'status': submission.status}
    db.session.add(submission)
    db.session.commit()
    app.logger.info('Submitted sample: {0} from IP: {1}'.format(s.sha256, ip_addr))
    proc = multiprocessing.Process(target=process_sample, args=(submission.id,))
    proc.start()
    # thread = threading.Thread(target=process_sample, args=(submission.id,))
    # thread.daemon = True
    # thread.start()
    # process_sample(submission.id)
    return json.dumps(return_data), 200


@app.route("/history")
def history():
    """
    Lists the history of submissions.
    """
    submissions = Submission.query.order_by(Submission.id.desc()).all()
    return_data = []
    for s in submissions:
        return_data.append({'id': s.id,
                            'sha256': s.sha256, 'time': str(s.time),
                            'classification': s.classification,
                            'possible_classification': s.possible_classification,
                            'ip_address': s.ip_address,
                            'status': s.status})
    return json.dumps(return_data), 200


@app.route("/classification/<sha256>")
def classification(sha256):
    """
    Provides the classifications for a specific hash.
    """
    submissions = Submission.query.filter_by(sha256=sha256.upper()).order_by(Submission.id.desc()).all()
    return_data = []
    for s in submissions:
        return_data.append({'id': s.id,
                            'sha256': s.sha256, 'time': str(s.time),
                            'classification': s.classification,
                            'possible_classification': s.possible_classification,
                            'ip_address': s.ip_address,
                            'status': s.status})
    return json.dumps(return_data), 200


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8888)