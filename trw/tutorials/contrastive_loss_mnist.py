import trw
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import trw.utils
from trw.train import make_pair_indices, LossContrastive
import torch.nn.functional as F


def export_scatter(embedding, output_truth, name, min_values=None, max_values=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = trw.train.make_unique_colors_f()

    if min_values is not None:
        assert max_values is not None
        ax.set_xlim(xmin=min_values[1], xmax=max_values[1])
        ax.set_ylim(ymin=min_values[0], ymax=max_values[0])

    for c in range(0, 10):
        c_indices = np.where(output_truth.squeeze(1) == c)
        x = embedding[c_indices][:, 1]
        y = embedding[c_indices][:, 0]
        ax.scatter(x, y, alpha=0.8, c=colors[c], edgecolors='none', s=10, label=str(c))

    plt.title(name)
    plt.legend(loc=2)
    plt.savefig(name)
    trw.train.export_figure(options.workflow_options.current_logging_directory, name)


class NetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)

        # project to a 2D embedding so the features can be visualized directly
        self.fc2 = nn.Linear(500, 2)

        # remove the bias to force the CNN to learn 0-centered features
        self.fc3 = nn.Linear(2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = trw.utils.flatten(x)
        fc1 = F.relu(self.fc1(x))

        # No ReLu: so that we can have negative features
        embedding = self.fc2(fc1)
        x = self.fc3(embedding)
        return x, embedding


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_embedding = NetEmbedding()

    def forward(self, batch):
        classes = batch['targets']
        indices_0, indices_1, same_target = make_pair_indices(classes)

        # a batch should be a dictionary of features
        x_0 = batch['images'][indices_0] / 255.0
        x_1 = batch['images'][indices_1] / 255.0

        f0, embedding = self.net_embedding(x_0)
        f1, _ = self.net_embedding(x_1)

        same_target = torch.tensor(same_target, dtype=torch.float32, device=x_0.device)
        contrastive_loss = LossContrastive()(f0, f1, same_target)

        return {
            'contrastive_loss': trw.train.OutputLoss(contrastive_loss),
            'embedding': trw.train.OutputEmbedding(embedding),
            'embedding_target': trw.train.OutputEmbedding(classes[indices_0]),
        }


# configure and run the training/evaluation
options = trw.train.Options(num_epochs=100)
trainer = trw.train.TrainerV2()

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(nb_workers=0),
    log_path='contrastive_loss_mnist',
    model=Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets=datasets,
        model=model,
        learning_rate=0.05)
)


splits = ['train', 'test']
max_values = None
min_values = None
for split in splits:
    outputs_values = results.outputs['mnist'][split]
    values = outputs_values['embedding']['output']
    if max_values is None:
        max_values = outputs_values['embedding']['output'].max(axis=0)
        min_values = outputs_values['embedding']['output'].min(axis=0)
        print("min:\n", min_values)
        print("max:\n", max_values)

    export_scatter(
        values,
        outputs_values['embedding_target']['output'],
        f'{split}_scatter',
        min_values=min_values,
        max_values=max_values
    )


while True:
    pass