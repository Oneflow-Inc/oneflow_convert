import os
import uuid
import oneflow as flow
from models import get_lenet_job_function

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
    temp_dir_name = str(uuid.uuid4())
    with open("lenet_qat_temp_dir_name.txt","w") as f:
        f.write(temp_dir_name)
    temp_dir = os.path.join("/tmp", temp_dir_name)
    flow.checkpoint.save(temp_dir)
