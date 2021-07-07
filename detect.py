import json
import shutil
import tempfile
import time
import cv2
import numpy as np
import tensorflow as tf
import valohai
import os
import zipfile

from yolov3_tf2.models import (
    YoloV3
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


params = {
    "size": 416,
    "production-pipeline": False
}

inputs = {
    "classes": ".data/classes.txt",
    "model": ".data/model/model.tf.data-00000-of-00001",
    "image": "data/street.jpg"
}

valohai.prepare(step="detect", default_parameters=params, default_inputs=inputs)


def main():
    model_path = tempfile.mkdtemp()

    with zipfile.ZipFile(valohai.inputs('model').path(process_archives=False), 'r') as zip_ref:
        zip_ref.extractall(model_path)

    with open(valohai.inputs('classes').path(), "r") as f:
        num_classes = len([line.strip("\n") for line in f if line != "\n"])
        print(f"Number of classes used: {num_classes}")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = YoloV3(classes=num_classes)

    weights_path = os.path.join(model_path, "model.tf")
    yolo.load_weights(weights_path).expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(valohai.inputs('classes').path()).readlines()]
    print(f'classes loaded ({class_names})')

    for path in valohai.inputs('image').paths():
        img_raw = tf.image.decode_image(open(path, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, valohai.parameters('size').value)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        basename = os.path.basename(path)
        output_path = valohai.outputs().path(f'output_{basename}')
        cv2.imwrite(output_path, img)
        print('output saved to: {}'.format(output_path))


    shutil.copyfile(
        valohai.inputs('model').path(process_archives=False),
        valohai.outputs("model").path("model.zip")
    )

    metadata = {
        "valohai.tags": ["model", "validated"],  # creates Valohai tags for the file
    }
    if valohai.parameters('production-pipeline').value:
        metadata["valohai.alias"] = ["production-model"]

    metadata_path = valohai.outputs("model").path('model.zip.metadata.json')
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)

if __name__ == "__main__":
    main()
