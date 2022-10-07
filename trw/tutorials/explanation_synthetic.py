import trw
import torch.nn as nn


def create_net_simple():
    activation = nn.ReLU
    n = trw.simple_layers.Input([None, 3, 64, 64], 'image')
    n = trw.simple_layers.convs_2d(n, channels=[4, 8, 16], activation=activation)
    n = trw.simple_layers.global_max_pooling_2d(n)
    n = trw.simple_layers.denses(n, sizes=[32, 2], activation=activation, last_layer_is_output=True)
    n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='triangle')
    return trw.simple_layers.compile_nn([n])


# configure and run the training/evaluation
trainer = trw.train.TrainerV2(callbacks_post_training=trw.train.default_post_training_callbacks(explain_decision=True))

results = trainer.fit(
    trw.train.Options(num_epochs=5),
    datasets=trw.datasets.create_fake_symbols_2d_dataset(
        image_shape=[64, 64],
        nb_classes_at_once=1,
        nb_samples=10000,
        global_scale_factor=0.3),
    log_path='explanation_synthetic2',
    model=create_net_simple(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_fn(datasets=datasets, model=model, learning_rate=0.001))

