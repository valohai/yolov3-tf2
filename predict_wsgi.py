import json
import os
import time
import zipfile
from functools import wraps

import numpy as np
import cv2

from werkzeug.debug import DebuggedApplication
from werkzeug.wrappers import Request, Response
import tensorflow as tf

from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import YoloV3


predictor = None

# root_path = "/tmp/yolo"
root_path = "/valohai/code"

def handle_valohai_prefix(environ):
    path = environ["PATH_INFO"]
    for prefix in (
        environ.get("HTTP_X_VH_PREFIX"),
        os.environ.get("VH_DEFAULT_PREFIX"),
    ):
        if not prefix:  # Could have no header or no envvar, so skip
            continue
        if path.startswith(prefix):  # If the path starts with this prefix,
            # ... then strip the prefix out as far as WSGI is concerned.
            environ["PATH_INFO"] = "/" + path[len(prefix) :].lstrip("/")
            break


def manage_prefixes(app):
    """
    Decorator to apply Valohai prefix management to a WSGI app callable.
    """

    @wraps(app)
    def prefix_managed_app(environ, start_response):
        handle_valohai_prefix(environ)
        return app(environ, start_response)

    return prefix_managed_app


def read_image_from_wsgi_request(environ):
    request = Request(environ)
    if not request.files:
        return None
    file_key = list(request.files.keys())[0]
    file = request.files.get(file_key)
    file.save(os.path.join(root_path, file.filename))
    img = tf.image.decode_image(open(os.path.join(root_path, file.filename), 'rb').read(), channels=3)
    return img


def predict_wsgi(environ, start_response):
    img_raw = read_image_from_wsgi_request(environ)

    num_classes = 5
    class_names = ['person', 'car', 'chair', 'book', 'bottle']
    size = 416

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=num_classes)

    if os.path.exists(f"{root_path}/model.zip"):
        with zipfile.ZipFile(f"{root_path}/model.zip", 'r') as zip_ref:
            zip_ref.extractall(root_path)
        os.remove(f"{root_path}/model.zip")

    weights_path = os.path.join(root_path, "model.tf")
    yolo.load_weights(weights_path).expect_partial()

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()

    box_count = int(nums[0])
    scores_by_class = {}

    for i in range(nums[0]):
        class_name = class_names[int(classes[0][i])]
        print('\t{}, {}, {}'.format(
            class_name,
            np.array(scores[0][i]),
            np.array(boxes[0][i])
        ))

        if class_name not in scores_by_class:
            scores_by_class[class_name] = {"count": 0, "score": 0.0, "boxes": []}

        scores_by_class[class_name]["score"] += float(scores[0][i])
        scores_by_class[class_name]["count"] += 1
        scores_by_class[class_name]["boxes"].append([float(coord) for coord in np.array(boxes[0][i])])

    speed = (t2 - t1)

    class_count = len(scores_by_class.keys())

    score_avg = sum([item["score"] / item["count"] for key, item in scores_by_class.items()]) / class_count if class_count > 0 else 0.0
    logs = dict(
        inference_speed=speed,
        score_avg=score_avg,
        rect_count=box_count,
    )
    logs.update({f"score_{key}": (item["score"] / item["count"] if item["count"] > 0 else 0.0) for key, item in scores_by_class.items()})
    print(json.dumps(dict(vh_metadata=logs)))

    if len(scores_by_class.keys()) > 0:
        response = Response(json.dumps(scores_by_class), mimetype='application/json')
    else:
        response = Response(json.dumps({"class": "None"}), mimetype='application/json')
    return response(environ, start_response)

predict_wsgi = DebuggedApplication(predict_wsgi)
predict_wsgi = manage_prefixes(predict_wsgi)


if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('localhost', 3000, predict_wsgi)
