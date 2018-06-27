from flask import Flask
from flask import render_template, abort, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.utils import secure_filename
from ...library.files import Sample
import os

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
    form = Submission()
    if form.validate_on_submit():
        f = form.sample.data
        s = Sample(frommemory=f.stream.read())
        filename = secure_filename(s.sha256)
        f.save(os.path.join('/samples', filename))
        return redirect(url_for('history'))
    return render_template('submit.html', form=form)


@app.route('/history')
def history():
    return render_template('history.html')


class Submission(FlaskForm):
    sample = FileField(validators=[FileRequired()])


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)