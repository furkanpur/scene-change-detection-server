import base64
import logging
import traceback

import cv2
import numpy as np
from cerberus import Validator
from flask import jsonify, request, Blueprint

from core.ai.scene_change.sc_detect_core import scene_change_detect
from schema.ai.sc_detect_schema import SC_DETECT_POST_REQUEST_SCHEMA

sc_detect_endpoint = Blueprint('sc_detect_endpoint', __name__)

logger = logging.getLogger(__name__)


@sc_detect_endpoint.route("/detect", methods=["POST"])
def sc_detect_endpoint_post_method():
    try:
        payload = request.json

        validator = Validator(SC_DETECT_POST_REQUEST_SCHEMA, allow_unknown=False)

        if not validator.validate(payload):
            error = {
                "info": {
                    "code": 10,
                    "message": str(validator.errors)
                }
            }

            logger.error(error)

            return jsonify(error), 400

        """ ------------------------------------------------------------------------------------ """
        """ IMAGE DECODE                                                                         """
        """ ------------------------------------------------------------------------------------ """
        old_image_str = payload['old_im']
        new_image_str = payload['new_im']

        old_image_b64 = base64.b64decode(old_image_str)
        # old_image_bytes = BytesIO(old_image_b64)
        # old_image_file = Image.open(old_image_bytes)

        old_image_bytes = np.fromstring(old_image_b64, dtype=np.uint8)
        old_image_file = cv2.imdecode(old_image_bytes, 1)

        new_image_b64 = base64.b64decode(new_image_str)
        # new_image_bytes = BytesIO(new_image_b64)
        # new_image_file = Image.open(new_image_bytes)

        new_image_bytes = np.fromstring(new_image_b64, dtype=np.uint8)
        new_image_file = cv2.imdecode(new_image_bytes, 1)

        """ ------------------------------------------------------------------------------------ """
        """ SCENE CHANGE DETECTION                                                               """
        """ ------------------------------------------------------------------------------------ """
        scene_change_result_b64 = scene_change_detect(old_image=old_image_file, new_image=new_image_file)

        response = {
            "info": {
                "code": 0,
                "message": "Success",
            },
            "data": {
                "result": scene_change_result_b64
            }
        }

        return jsonify(response), 200

    except Exception:
        error = {
            "info": {
                "code": 99,
                "message": "Unknown error."
            }
        }

        logger.error(traceback.format_exc())
        logger.error(error)

        return jsonify(error), 500
