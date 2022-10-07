from torchvision import models
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import trw


def preprocess_image(pil_im):
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H

    preprocessed = (im_as_arr / 255 - np.asarray(mean).reshape([3, 1, 1])) / np.asarray(std).reshape([3, 1, 1])
    im_as_ten = torch.from_numpy(preprocessed).float()
    return im_as_ten


def input_dataset():
    example_list = [
        'input_images/boat.jpg',
        'input_images/snake.jpg',
        'input_images/cat_dog.png',
        'input_images/spider.png',
    ]

    images = []
    fake_ids = []
    for fake_id, path in enumerate(example_list):
        image = Image.open(path).convert('RGB')
        image = preprocess_image(image)

        images.append(image)
        fake_ids.append(fake_id)

    images = torch.from_numpy(np.stack(images))
    fake_ids = torch.from_numpy(np.stack(fake_ids).astype(np.int64))

    return {
        'imagenet': {
            'valid': trw.train.SequenceArray({
                'images': images,
                'ids': fake_ids
            }, sampler=trw.train.SamplerSequential(batch_size=100))
        }
    }


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)

    def forward(self, batch):
        images = batch['images']
        output = self.model(images)
        return {
            'output': trw.train.OutputClassification(output, batch['ids'], classes_name='ids')
        }


options = trw.train.Options(num_epochs=0)
trainer = trw.train.TrainerV2(
    callbacks_pre_training=None,
    trainer_callbacks_per_batch=None,
    callbacks_post_training=[trw.callbacks.CallbackExplainDecision()]
)

results = trainer.fit(
    options,
    datasets=input_dataset(),
    log_path='imagenet-explanation',
    model=Model(),
    optimizers_fn=None)
