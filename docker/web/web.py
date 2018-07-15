from flask import Flask, render_template, abort, redirect, url_for, flash, request
from urllib.parse import urlparse, urljoin
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
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
from functools import wraps
sys.path.append('..')
sys.path.append(os.path.join('..', '..'))
from library.files import Sample
from docker.db_models.models import Submission, WebRequest, User, setup_database, db, generate_api_key
from ..common.token import generate_confirmation_token, confirm_token
from ..common.email import mail, send_email


# Global values
API_URL = "http://api:8888"
API_KEY_LENGTH_BYTES = 35
POSSIBLE_CLASSIFICATIONS = [
    ('Trojan', 'Trojan'),
    ('Virus', 'Virus'),
    ('Worm', 'Worm'),
    ('Backdoor', 'Backdoor'),
    ('Ransomware', 'Ransomware'),
    ('PUA', 'PUA'),
    ('Unknown', 'Unknown')
]
MULTIUSER = bool(int(os.environ['MULTIUSER']))
HISTORY_LENGTH = int(os.environ['HISTORY_LENGTH'])

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


def get_userid():
    user_id = flask_login.current_user.id
    return user_id


if MULTIUSER:
    limiter = Limiter(
        app,
        key_func=get_userid
    )
else:
    limiter = None


def limit_api_resets(f):
    """ Decorates functions depending on multiuser mode. """
    if MULTIUSER:
        @limiter.limit("1 per minute")
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    else:
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper


@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow()}


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def get_request_ip():
    """
    Gets the requester IP.
    :return: The IP.
    """
    return request.headers.get('X-Forwarded-For', request.environ['REMOTE_ADDR'])


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
    email = StringField('Email', id="email", validators=[DataRequired(), Email()],
                        render_kw={"placeholder": "email@example.com"})
    password = PasswordField('Password', id="password", validators=[DataRequired()],
                             render_kw={"placeholder": "Password"})


class RegisterForm(FlaskForm):
    """
    The register form.
    """
    email = StringField('Email', validators=[DataRequired(), Email()], render_kw={"placeholder": "email@example.com"})
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=64),
                                                     EqualTo('password_confirm', message='Passwords must match')],
                             render_kw={"placeholder": "Password"})
    password_confirm = PasswordField('Confirm Password', render_kw={"placeholder": "Password"})


class EmailOnlyForm(FlaskForm):
    """
    The email only form.
    """
    email = StringField('Email', validators=[DataRequired(), Email()], render_kw={"placeholder": "email@example.com"})


class PasswordOnlyForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=64),
                                                     EqualTo('password_confirm', message='Passwords must match')],
                             render_kw={"placeholder": "Password"})
    password_confirm = PasswordField('Confirm Password', render_kw={"placeholder": "Confirm"})


class PasswordChangeForm(FlaskForm):
    old_password = PasswordField('Old Password', render_kw={"placeholder": "Old Password"})
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=8, max=64),
                                                         EqualTo('password_confirm', message='Passwords must match')],
                             render_kw={"placeholder": "New Password"})
    password_confirm = PasswordField('New Confirm Password', render_kw={"placeholder": "Confirm"})


class StateInfo(object):
    def __init__(self, data):
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary")
        self.__dict__ = data


def flash_errors(form, flash_type='danger'):
    """ Flashes form errors. """
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error in the {0} field - {1}".format(getattr(form, field).label.text, error), flash_type)


def login_decorate(somefunc):
    """ Decorates functions depending on multiuser mode. """
    if MULTIUSER:
        return flask_login.login_required(somefunc)
    else:
        return somefunc


@app.route('/')
def main():
    """
    The main page.
    """
    ip_addr = get_request_ip()
    State = {'multiuser': MULTIUSER}
    return render_template('main.html', state=StateInfo(State))


@app.route('/login', methods=('GET', 'POST'))
def login():
    """
    The login page.
    """
    ip_addr = get_request_ip()
    form = LoginForm()
    if request.method == 'POST':
        if form.validate():
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.activated and user.validate_password(form.password.data):
                flask_login.login_user(user)
                user.last_login = datetime.datetime.now()
                user.last_login_ip = ip_addr
                db.session.add(user)
                db.session.commit()
                flash("Successfully logged in.", 'success')
                app.logger.info('User: {0} ID: {1} successful login from IP: {2}'.format(form.email.data, user.id, ip_addr))
                # next = request.args.get('next')
                # # is_safe_url should check if the url is safe for redirects.
                # # See http://flask.pocoo.org/snippets/62/ for an example.
                # if not is_safe_url(next):
                #     return abort(400)
                # if next:
                #     return redirect(next)
                return redirect(url_for('main'))
            elif not user.activated:
                flash("Not yet activated.  Check your email and click the activation link!", 'danger')
                app.logger.info('User: {0} ID: {1} not activated yet from IP: {2}'.format(form.email.data,
                                                                                          user.id, ip_addr))
            else:
                flash("Invalid login.  Try again.", 'danger')
                app.logger.info('User: {0} failed login from IP: {1}'.format(form.email.data, ip_addr))
                return redirect(url_for('login'))
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(u"Error in the {0} field - {1}".format(getattr(form, field).label.text, error), 'danger')
            app.logger.info('User: {0} failed login from IP: {1}'.format(form.email.data, ip_addr))
            return redirect(url_for('register'))
    State = {'multiuser': MULTIUSER}
    return render_template('login.html', state=StateInfo(State), form=form)


@app.route('/logout')
@login_decorate
def logout():
    ip_addr = get_request_ip()
    app.logger.info('User: {0} ID: {1} logout from IP: {2}'.format(flask_login.current_user.email,
                                                                   flask_login.current_user.id, ip_addr))
    flask_login.logout_user()
    return redirect(url_for('main'))


@app.route('/register', methods=('GET', 'POST'))
def register():
    ip_addr = get_request_ip()
    form = RegisterForm()
    if request.method == 'POST':
        if form.validate():
            users = User.query.filter_by(email=form.email.data).all()
            if len(users) == 0:
                api_key = generate_api_key(API_KEY_LENGTH_BYTES)
                user = User(email=form.email.data, password=form.password.data, registration=datetime.datetime.now(),
                            api_key=api_key)
                token = generate_confirmation_token(user.email)
                confirm_url = url_for('confirm', token=token, _external=True)
                html = render_template('activationemail.html', confirm_url=confirm_url)
                subject = "Please confirm your email"
                send_email(user.email, subject, html)
                db.session.add(user)
                db.session.commit()
                flash("Account registered.  An activation email was sent to you for further instructions.", 'success')
                app.logger.info('User: {0} ID: {1} registered from IP: {2}'.format(user.email, user.id, ip_addr))
                return redirect(url_for('login'))
            else:
                flash("Email already registered.", 'danger')
                app.logger.info('User: {0} already registered from IP: {1}'.format(form.email.data, ip_addr))
                return redirect(url_for('register'))
        else:
            flash_errors(form)
            return redirect(url_for('register'))
    State = {'multiuser': MULTIUSER}
    return render_template('register.html', state=StateInfo(State), form=form)


@app.route('/confirm/<token>')
# @login_decorate
def confirm(token):
    ip_addr = get_request_ip()
    try:
        email = confirm_token(token)
    except:
        flash('The confirmation link is invalid or has expired.', 'danger')
        app.logger.info('Confirmation link dead from IP: {0}'.format(ip_addr))
        return redirect(url_for('login'))
    if not email:
        flash('The confirmation link is invalid or has expired.', 'danger')
        app.logger.info('Confirmation link dead from IP: {0}'.format(ip_addr))
        return redirect(url_for('login'))
    user = User.query.filter_by(email=email).first_or_404()
    if user.activated:
        flash('Account already confirmed.', 'success')
        app.logger.info('User: {0} ID: {1} already activated from IP: {2}'.format(flask_login.current_user.email,
                                                                                  flask_login.current_user.id, ip_addr))
    else:
        user.activated = True
        user.activated_date = datetime.datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')
        app.logger.info('User: {0} ID: {1} confirmed from IP: {2}'.format(user.email, user.id, ip_addr))
    return redirect(url_for('main'))


@app.route('/resend_activation_email', methods=('GET', 'POST'))
def resend_activation_email():
    ip_addr = get_request_ip()
    form = EmailOnlyForm()
    if request.method == 'POST':
        if form.validate():
            user = User.query.filter_by(email=form.email.data).first()
            flash('If a new registration exists for this email address, a new activation email was sent.', 'success')
            if user and user.email and not user.activated:
                token = generate_confirmation_token(user.email)
                confirm_url = url_for('confirm', token=token, _external=True)
                html = render_template('activationemail.html', confirm_url=confirm_url)
                subject = "Please confirm your email"
                send_email(user.email, subject, html)
                app.logger.info('User: {0} ID: {1} resending activation email from IP: {2}'.format(user.email,
                                                                                                   user.id, ip_addr))
            return redirect(url_for('main'))
        else:
            flash_errors(form)
            return redirect(url_for('resend_activation_email'))
    State = {'multiuser': MULTIUSER}
    return render_template('resend_activation_email.html', state=StateInfo(State), form=form)


@app.route('/forgot_password', methods=('GET', 'POST'))
def forgot_password():
    ip_addr = get_request_ip()
    form = EmailOnlyForm()
    if request.method == 'POST':
        if form.validate():
            user = User.query.filter_by(email=form.email.data).first()
            flash('If an account exists for this email address, an email with further instructions was sent.', 'success')
            if user and user.email and user.activated:
                token = generate_confirmation_token(user.email)
                password_reset_url = url_for('reset_password', token=token, _external=True)
                html = render_template('forgotpasswordemail.html', password_reset_url=password_reset_url)
                subject = "Reset your password"
                send_email(user.email, subject, html)
                app.logger.info('User: {0} ID: {1} sending forgotten password email from IP: {2}'.format(user.email,
                                                                                                         user.id,
                                                                                                         ip_addr))
            return redirect(url_for('main'))
        else:
            flash_errors(form)
            return redirect(url_for('forgot_password'))
    State = {'multiuser': MULTIUSER}
    return render_template('forgot_password.html', state=StateInfo(State), form=form)


@app.route('/reset_password/<token>', methods=('GET', 'POST'))
def reset_password(token):
    ip_addr = get_request_ip()
    try:
        email = confirm_token(token)
    except:
        flash('The password reset link is invalid or has expired.', 'danger')
        app.logger.info('Password reset link dead from IP: {0}'.format(ip_addr))
        return redirect(url_for('login'))
    if not email:
        flash('The password reset link is invalid or has expired.', 'danger')
        app.logger.info('Password reset link dead from IP: {0}'.format(ip_addr))
        return redirect(url_for('login'))
    form = PasswordOnlyForm()
    user = User.query.filter_by(email=email).first_or_404()
    State = {'multiuser': MULTIUSER}
    if request.method == 'GET':
        if user.activated:
            app.logger.info('User: {0} ID: {1} reset password from IP: {2}'.format(user.email,
                                                                                   user.id, ip_addr))
            return render_template('reset_password.html', state=StateInfo(State), form=form, token=token)
        else:
            flash('Your account has not been activated yet.  Resend the activation email first!', 'danger')
            app.logger.info('User: {0} ID: {1} reset password but not confirmed from IP: {2}'.format(user.email,
                                                                                                     user.id, ip_addr))
            return redirect(url_for('main'))
    elif request.method == 'POST':
        if form.validate():
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Password has been reset.', 'success')
            app.logger.info('User: {0} ID: {1} reset password successfully from IP: {2}'.format(user.email,
                                                                                                user.id, ip_addr))
            return redirect('login')
        else:
            flash_errors(form)
            return redirect(url_for('reset_password', token=token))


@app.route('/set_password', methods=('GET', 'POST'))
@login_decorate
def set_password():
    ip_addr = get_request_ip()
    form = PasswordChangeForm()
    State = {'multiuser': MULTIUSER}
    user = flask_login.current_user
    if request.method == 'GET':
        return render_template('set_password.html', form=form, state=StateInfo(State))
    elif request.method == 'POST':
        if form.validate():
            if user.validate_password(form.old_password.data):
                user.set_password(form.password.data)
                db.session.add(user)
                db.session.commit()
                app.logger.info('User: {0} ID: {1} set password successfully from IP: {2}'.format(user.email,
                                                                                                  user.id, ip_addr))
                flash('Password has been set.', 'success')
            return redirect('myaccount')
        else:
            flash_errors(form)
            return redirect(url_for('set_password'))


@app.route('/set_email', methods=('GET', 'POST'))
@login_decorate
def set_email():
    ip_addr = get_request_ip()
    form = EmailOnlyForm()
    State = {'multiuser': MULTIUSER}
    user = flask_login.current_user
    if request.method == 'GET':
        return render_template('set_email.html', form=form, state=StateInfo(State))
    elif request.method == 'POST':
        if form.validate():
            users = User.query.filter_by(email=form.email.data).all()
            if len(users) == 0:
                user.email = form.email.data.strip()
                user.activated = False
                token = generate_confirmation_token(user.email)
                confirm_url = url_for('confirm', token=token, _external=True)
                html = render_template('activationemail.html', confirm_url=confirm_url)
                subject = "Please confirm your new email"
                send_email(user.email, subject, html)
                db.session.add(user)
                db.session.commit()
                flash("Email set.  An activation email was sent to you for further instructions before "
                      "you can use this system.", 'success')
                app.logger.info('User: {0} ID: {1} changed email from IP: {2}'.format(user.email, user.id, ip_addr))
                return redirect(url_for('main'))
            else:
                flash("Email already registered.", 'danger')
                app.logger.info('User: {0} ID: {1} changed email but already a registered email from IP: {2}'.format(user.email,
                                                                                                                     user.id,
                                                                                                                     ip_addr))
                return redirect(url_for('set_email'))
            return redirect('myaccount')
        else:
            flash_errors(form)
            return redirect(url_for('set_email'))


@app.route('/generate_new_api_key')
@login_decorate
@limit_api_resets
def generate_new_api_key():
    user = flask_login.current_user
    api_key = generate_api_key(API_KEY_LENGTH_BYTES)
    user_check = User.query.filter_by(api_key=api_key).count()
    while user_check > 0:
        api_key = generate_api_key(API_KEY_LENGTH_BYTES)
    user.api_key = api_key
    db.session.add(user)
    db.session.commit()
    return redirect(url_for('myaccount'))


@app.route('/myaccount')
@login_decorate
def myaccount():
    ip_addr = get_request_ip()
    State = {'multiuser': MULTIUSER}
    return render_template('myaccount.html', state=StateInfo(State), user=flask_login.current_user)


@app.route('/submit', methods=('GET', 'POST'))
@login_decorate
def submit():
    """
    The sample submission page.
    """
    ip_addr = get_request_ip()
    form = SubmissionForm()
    user = flask_login.current_user
    if form.validate_on_submit():
        f = form.sample.data
        s = Sample(frommemory=f.stream.read())
        files = {'file': s.rawdata}
        data = {'classification': form.classification.data}
        url = "{0}/submit".format(API_URL)
        if MULTIUSER:
            url = url+"?apikey={0}".format(user.api_key)
        try:
            req = requests.post(url, files=files, data=data)
        except Exception as exc:
            flash('Exception while sending file to API!', 'danger')
            if MULTIUSER:
                app.logger.exception('Error submitting sample: {0} from User: {1} ID: {2} IP: {3} - Exception: {4}'.format(
                    s.sha256, user.email, user.id, ip_addr, exc))
            else:
                app.logger.exception('Error submitting sample: {0} from IP: {1} - Exception: {2}'.format(s.sha256,
                                                                                                         ip_addr, exc))
            return redirect(url_for('submit'))
        if req.status_code == 429:
            flash("You exceeded your rate limit of {0}.  Please wait a bit.".format(user.api_limits), 'danger')
            return redirect(url_for('main'))
        elif req.status_code != 200:
            flash("API FAILURE - HTTP Code: {0}".format(req.status_code), 'danger')
            app.logger.error('Submit API did not return 200: {0}'.format(req))
            return redirect(url_for('submit'))
        if MULTIUSER:
            user_id = flask_login.current_user.id
        else:
            user_id = None
        submit_time = datetime.datetime.now()
        req = WebRequest(sha256=s.sha256, time=submit_time,
                         possible_classification=form.classification.data,
                         ip_address=ip_addr, user_id=user_id)
        db.session.add(req)
        db.session.commit()
        if MULTIUSER:
            app.logger.info('Submitted sample: {0} from User: {1} ID: {2} IP: {3}'.format(s.sha256, user.email,
                                                                                          user.id, ip_addr))
        else:
            app.logger.info('Submitted sample: {0} from IP: {1}'.format(s.sha256, ip_addr))
        return redirect(url_for('history'))
    else:
        flash_errors(form)
    State = {'multiuser': MULTIUSER}
    return render_template('submit.html', state=StateInfo(State), form=form)


@app.route('/history')
@login_decorate
def history():
    """
    The submission history page.
    """
    ip_addr = get_request_ip()
    user = flask_login.current_user
    url = "{0}/history".format(API_URL)
    if MULTIUSER:
        url = url + "?apikey={0}".format(user.api_key)
    try:
        req = requests.get(url)
    except Exception as exc:
        flash('Exception while pulling history from API!', 'danger')
        app.logger.exception('Exception while pulling history - Exception: {0}'.format(exc))
        return redirect(url_for('main'))
    if req.status_code == 429:
            flash("You exceeded your rate limit of {0}.  Please wait a bit.".format(user.api_limits), 'danger')
            return redirect(url_for('main'))
    elif req.status_code != 200:
        flash("API FAILURE - HTTP Code: {0}".format(req.status_code), 'danger')
        app.logger.error('History API did not return 200: {0}'.format(req))
        return redirect(url_for('main'))
    history = json.loads(req.text)
    history = history[:HISTORY_LENGTH]
    State = {'multiuser': MULTIUSER}
    return render_template('history.html', state=StateInfo(State), history=history)


@app.route('/info_api')
@login_decorate
def info_api():
    """
    The API information page.
    """
    ip_addr = get_request_ip()
    State = {'multiuser': MULTIUSER, 'hostname': request.host}
    return render_template('api.html', state=StateInfo(State))


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)
