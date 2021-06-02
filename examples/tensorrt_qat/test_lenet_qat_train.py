import os
import shutil
import uuid
import argparse
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
