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
import oneflow as flow
from models import get_lenet_job_function, LENET_MODEL_QAT_DIR


if __name__ == "__main__":
    batch_size = 100
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        batch_size, batch_size
    )
    # train
    train_job = get_lenet_job_function("train", batch_size=batch_size)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
    if os.path.exists(LENET_MODEL_QAT_DIR):
        shutil.rmtree(LENET_MODEL_QAT_DIR)
    flow.checkpoint.save(LENET_MODEL_QAT_DIR)
    # Without the following 'print' CI won't pass, but I have no idea why.
    print(
        "Model was saved at "
        + LENET_MODEL_QAT_DIR
        + ". Status : "
        + str(os.path.exists(LENET_MODEL_QAT_DIR))
    )
