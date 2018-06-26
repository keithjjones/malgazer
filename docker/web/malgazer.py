from flask import Flask
from flask import render_template
from flask_material import Material


app = Flask(__name__)
Material(app)

@app.route('/')
def main():
    return render_template('main.html')


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)