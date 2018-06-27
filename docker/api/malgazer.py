from flask import Flask, flash, request, redirect, url_for, render_template
import json
import os
from werkzeug.utils import secure_filename
from ...library.files import Sample


app = Flask(__name__)


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
    filepath = os.path.join('/samples', filename)
    if not os.path.isfile(filepath):
        with open(filepath, 'wb') as f_out:
            f_out.write(s.rawdata)
    return_data = {'sha256': s.sha256}
    return json.dumps(return_data)


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)