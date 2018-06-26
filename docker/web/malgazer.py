from flask import Flask
from flask import render_template
from flask_material import Material


app = Flask(__name__)
Material(app)

@app.route('/')
def main():
    return render_template('main.html')


@app.route('/submit')
def submit():
    return render_template('submit.html')


@app.route('/history')
def history():
    return render_template('history.html')


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)