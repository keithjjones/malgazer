from flask import Flask, render_template, abort, redirect, url_for, flash, request
from urllib.parse import urlparse, urljoin
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from wtforms import RadioField, StringField, PasswordField, ValidationError
from wtforms.validators import DataRequired, Email, EqualTo, Length
from flask_wtf.file import FileField, FileRequired
from flask_wtf.csrf import CSRFProtect, CSRFError
import flask_login
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
from docker.db_models.models import Submission, WebRequest, User, setup_database, db
from ..common.token import generate_confirmation_token, confirm_token
from ..common.email import mail, send_email


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
app.config['MAIL_SERVER'] = os.environ['MAIL_SERVER']
app.config['MAIL_USE_SSL'] = bool(int(os.environ['MAIL_USE_SSL']))
app.config['MAIL_USE_TLS'] = bool(int(os.environ['MAIL_USE_TLS']))
app.config['MAIL_PORT'] = int(os.environ['MAIL_PORT'])
app.config['MAIL_USERNAME'] = os.environ['MAIL_USERNAME']
app.config['MAIL_PASSWORD'] = os.environ['MAIL_PASSWORD']


db.init_app(app)
mail.init_app(app)
csrf = CSRFProtect(app)
applogger = app.logger
file_handler = logging.handlers.TimedRotatingFileHandler("/logs/web.log", when='midnight', backupCount=30)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
applogger.setLevel(logging.DEBUG)
applogger.addHandler(file_handler)
login_manager = flask_login.LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc


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


class RegisterForm(FlaskForm):
    """
    The register form.
    """
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=64),
                                                     EqualTo('password_confirm', message='Passwords must match')])
    password_confirm = PasswordField('Confirm Password')


class StateInfo(object):
    def __init__(self, data):
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary")
        self.__dict__ = data


def login_decorate(somefunc):
    if MULTIUSER:
        return flask_login.login_required(somefunc)
    else:
        return somefunc


@app.route('/')
def main():
    """
    The main page.
    """
    State = {'multiuser': MULTIUSER}
    return render_template('main.html', state=StateInfo(State))


@app.route('/login', methods=('GET', 'POST'))
def login():
    """
    The login page.
    """
    ip_addr = request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])
    form = LoginForm()
    if request.method == 'POST':
        if form.validate():
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.validate_password(form.password.data):
                flask_login.login_user(user)
                flash("Successfully logged in.", 'success')

                # next = request.args.get('next')
                # # is_safe_url should check if the url is safe for redirects.
                # # See http://flask.pocoo.org/snippets/62/ for an example.
                # if not is_safe_url(next):
                #     return abort(400)
                # if next:
                #     return redirect(next)
                return redirect(url_for('main'))
            else:
                flash("Invalid login.  Try again.", 'danger')
                return redirect(url_for('login'))
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(u"Error in the {0} field - {1}".format(getattr(form, field).label.text, error), 'danger')
            return redirect(url_for('register'))
    State = {'multiuser': MULTIUSER}
    return render_template('login.html', state=StateInfo(State), form=form)


@app.route('/logout')
@login_decorate
def logout():
    flask_login.logout_user()
    return redirect(url_for('main'))


@app.route('/register', methods=('GET', 'POST'))
def register():
    form = RegisterForm()
    if request.method == 'POST':
        if form.validate():
            users = User.query.filter_by(email=form.email.data).all()
            if len(users) == 0:
                user = User(email=form.email.data, password=form.password.data, registration=datetime.datetime.now())
                token = generate_confirmation_token(user.email)
                confirm_url = url_for('confirm', token=token, _external=True)
                html = render_template('activationemail.html', confirm_url=confirm_url)
                subject = "Please confirm your email"
                send_email(user.email, subject, html)
                db.session.add(user)
                db.session.commit()
                flash("Account registered.  An activation email was sent to you for further instructions.", 'success')
                return redirect(url_for('login'))
            else:
                flash("Email already registered.", 'danger')
                return redirect(url_for('register'))
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(u"Error in the {0} field - {1}".format(getattr(form, field).label.text, error), 'danger')
            return redirect(url_for('register'))
    State = {'multiuser': MULTIUSER}
    return render_template('register.html', state=StateInfo(State), form=form)


@app.route('/confirm/<token>')
@login_decorate
def confirm(token):
    try:
        email = confirm_token(token)
    except:
        flash('The confirmation link is invalid or has expired.', 'danger')
    user = User.query.filter_by(email=email).first_or_404()
    if user.activated:
        flash('Account already confirmed. Please login.', 'success')
    else:
        user.activated = True
        user.activated_date = datetime.datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')
    return redirect(url_for('main'))


@app.route('/myaccount')
@login_decorate
def myaccount():
    State = {'multiuser': MULTIUSER}
    return render_template('myaccount.html', state=StateInfo(State))


@app.route('/submit', methods=('GET', 'POST'))
@login_decorate
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
            flash('Exception while sending file to API!', 'danger')
            app.logger.exception('Error submitting sample: {0} - Exception: {1}'.format(s.sha256, exc))
            return redirect(url_for('submit'))
        if req.status_code != 200:
            flash("API FAILURE - HTTP Code: {0}".format(req.status_code), 'danger')
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
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(u"Error in the {0} field - {1}".format(getattr(form, field).label.text, error), 'danger')
    State = {'multiuser': MULTIUSER}
    return render_template('submit.html', state=StateInfo(State), form=form)


@app.route('/history')
@login_decorate
def history():
    """
    The submission history page.
    """
    url = "{0}/history".format(API_URL)
    try:
        req = requests.get(url)
    except Exception as exc:
        flash('Exception while pulling history from API!', 'danger')
        app.logger.exception('Exception while pulling history - Exception: {0}'.format(exc))
        return redirect(url_for('main'))
    if req.status_code != 200:
        flash("API FAILURE - HTTP Code: {0}".format(req.status_code), 'danger')
        app.logger.error('History API did not return 200: {0}'.format(req))
        return redirect(url_for('main'))
    history = json.loads(req.text)
    State = {'multiuser': MULTIUSER}
    return render_template('history.html', state=StateInfo(State), history=history)


@app.route('/api')
@login_decorate
def api():
    """
    The API information page.
    """
    State = {'multiuser': MULTIUSER}
    return render_template('api.html', state=StateInfo(State))


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
