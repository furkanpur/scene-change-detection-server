import logging

from flask import Flask
from flask_cors import CORS

from controller.ai.scene_change.sc_detect_controller import sc_detect_endpoint

app = Flask(__name__)

CORS(app)

app.register_blueprint(sc_detect_endpoint, url_prefix="/ai")

logger = logging.getLogger(__name__)


def run():
    app.run(host="0.0.0.0")


if __name__ == "__main__":
    run()
