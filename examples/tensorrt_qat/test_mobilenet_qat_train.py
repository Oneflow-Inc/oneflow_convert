"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import shutil
import cv2
import numpy as np
import oneflow as flow
from models import get_mobilenet_job_function, MOBILENET_MODEL_QAT_DIR


def resize(images):
    results = []
    for image in images:
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (224, 224))
        results.append(image[None, :, :])
    return np.array(results)


if __name__ == "__main__":
    batch_size = 16
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        batch_size, batch_size
    )
    # train
    train_job = get_mobilenet_job_function("train", batch_size=batch_size)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            images = resize(images)
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
            if i == 100:
                break
    if os.path.exists(MOBILENET_MODEL_QAT_DIR):
        shutil.rmtree(MOBILENET_MODEL_QAT_DIR)
    flow.checkpoint.save(MOBILENET_MODEL_QAT_DIR)
    # Without the following 'print' CI won't pass, but I have no idea why.
    print(
        "Model was saved at "
        + MOBILENET_MODEL_QAT_DIR
        + ". Status : "
        + str(os.path.exists(MOBILENET_MODEL_QAT_DIR))
    )
