import trw
import os


def create_net(hparams):
    # here we use the `trw.hparams.HyperParameterRepository.current_hparams` instance
    # for ease of use.
    number_hidden = hparams.create(trw.hparams.DiscreteInteger('number_hidden', 500, 100, 1000))
    number_conv1_channels = hparams.create(trw.hparams.DiscreteInteger('number_conv1_channels', 16, 4, 64))

    n = trw.simple_layers.Input([None, 1, 28, 28], 'images')
    n = trw.simple_layers.Conv2d(n, out_channels=number_conv1_channels, kernel_size=5, stride=2)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.MaxPool2d(n, 2, 2)
    n = trw.simple_layers.Flatten(n)
    n = trw.simple_layers.Linear(n, number_hidden)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.Linear(n, 10)
    n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
    return trw.simple_layers.compile_nn([n])
    

def evaluate_hparams(hparams):
    # resort to `trw.hparams.HyperParameterRepository.current_hparams`. It is the same
    # object as `hparams`.
    learning_rate = hparams.create(trw.hparams.ContinuousPower('learning_rate', 0.1, -5, -1))

    # disable most of the reporting so that we don't end up with
    # thousands of files that are not useful for hyper-parameter
    # search
    trainer = trw.train.TrainerV2(
        callbacks_pre_training=None,
        callbacks_post_training=None,
        callbacks_per_epoch=[trw.callbacks.callback_epoch_summary.CallbackEpochSummary()]
    )
    
    results = trainer.fit(
        options,
        datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True),
        log_path='run',
        model=create_net(hparams),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
            datasets=datasets,
            model=model,
            learning_rate=learning_rate)
    )
    
    hparam_loss = trw.utils.to_value(results.history[-1]['mnist']['test']['overall_loss']['loss'])
    return {'loss': hparam_loss}, results.history, 'additional info goes here!'
    
    
# configure and run the training/evaluation
options = trw.train.Options(num_epochs=5)
hparams_root = os.path.join(options.workflow_options.logging_directory, 'mnist_cnn_hparams')
trw.train.utilities.create_or_recreate_folder(hparams_root)
options.workflow_options.logging_directory = hparams_root

# run the hyper-parameter search
random_search = trw.hparams.HyperParametersOptimizerRandomSearchLocal(
    evaluate_fn=evaluate_hparams,
    repeat=40)

store = trw.hparams.RunStoreFile(os.path.join(hparams_root, 'hparams.pkl'))
runs = random_search.optimize(store)

# finally analyse the run
trw.hparams.analyse_hyperparameters(
    run_results=runs,
    output_path=hparams_root)
