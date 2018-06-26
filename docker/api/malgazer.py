from flask import Flask
from flask import render_template
import json
app = Flask(__name__)


@app.route('/')
def main():
    return json.dumps({})


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8080)