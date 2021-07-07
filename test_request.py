import random
import time
import requests
import valohai

params = {
    "endpoint": "https://valohai.cloud/valohai/yolov3-tf2/yolov3/production/predict",
    "min_seconds": 60,
    "max_seconds": 600,
}

inputs = {
    "images": "datum://017a71bc-5e27-e0d2-a2ff-676ca416bcf1"
}

url = valohai.parameters("endpoint").value
logger = valohai.logger()

valohai.prepare(step="request generator", default_parameters=params, default_inputs=inputs, image="")

while True:
    file_name = random.choice(valohai.inputs('images')._get_input_vfs().files).name
    file_path = valohai.inputs('images').path(file_name)

    print(f"uploading {file_path}")
    files = {'file': open(file_path,'rb')}
    r = requests.post(url, files=files)

    logger.log("status", r.status_code)
    logger.flush()

    time.sleep(random.randrange(valohai.parameters('min_seconds').value, valohai.parameters('max_seconds').value))
