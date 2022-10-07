import trw
import torch.nn as nn
import torch.nn.functional as F
import os


class Net_simple(nn.Module):
    def __init__(self, options):
        super().__init__()

        dropout_probability = options.training_parameters['dropout_probability']
        self.convs = trw.layers.convs_2d(3, [64, 128, 256], batch_norm_kwargs={}, convolution_repeats=[3, 3, 3])
        self.denses = trw.layers.denses([4096, 256, 10], dropout_probability=None, batch_norm_kwargs={}, last_layer_is_output=True)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images']
        x = self.convs(x)
        x = self.denses(x)

        return {
            'softmax': trw.train.OutputClassification(x, batch['targets'], classes_name='targets')
        }


class Net_DARTS(nn.Module):
    def __init__(self, options):
        super().__init__()

        c = 16
        stem_multiplier = 3
        multiplier = 4
        layers = 8


        c_curr = stem_multiplier * c  # 3*16
        self.stem = nn.Sequential(  # 3 => 48
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr)
        )

        cpp, cp, c_curr = c_curr, c_curr, c  # 48, 48, 16
        self.cells = nn.ModuleList()
        reduction_prev = False

        weights_reduction = None
        weights_normal = None
        for i in range(layers):

            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
            # the output channels = multiplier * c_curr
            if reduction:
                # here we share weights for all normal cells and another set of weights for all reduction cells
                weights = weights_reduction
            else:
                weights = weights_normal

            cell = trw.arch.Cell(
                primitives=trw.arch.DARTS_PRIMITIVES_2D,
                cpp=cpp,
                cp=cp,
                c=c_curr,
                is_reduction=reduction,
                is_reduction_prev=reduction_prev,
                weights=weights)

            if weights is None:
                if reduction:
                    weights_reduction = cell.get_weights()
                    self.cell_reduction = cell
                else:
                    weights_normal = cell.get_weights()
                    self.cell_normal = cell

            # update reduction_prev
            reduction_prev = reduction

            self.cells += [cell]

            cpp, cp = cp, multiplier * c_curr

        # adaptive pooling output size to 1x1
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        self.classifier = nn.Linear(cp, 10)

        self.weights_dict = {
            # dict to prevent auto added parameter to the model parameters to
            'weights_reduction': weights_reduction,
            'weights_normal': weights_normal,
        }

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images']

        s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
        for i, cell in enumerate(self.cells):
            # weights are shared across all reduction cell or normal cell
            # according to current cell's type, it choose which architecture parameters
            # to use
            #weights = cell.get_weights() # [14, 8]
            # execute cell() firstly and then assign s0=s1, s1=result
            s0, s1 = s1, cell([s0, s1]) # [40, 64, 32, 32]

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return {
            'softmax': trw.train.OutputClassification(logits, batch['targets'], classes_name='targets')
        }
    
    def export_configuration(self, path):
        with open(os.path.join(path, 'genotype_normal.txt'), 'w') as f:
            f.write(str(self.cell_normal.get_genotype()))
            
        with open(os.path.join(path, 'genotype_reduction.txt'), 'w') as f:
            f.write(str(self.cell_reduction.get_genotype()))


if __name__ == '__main__':
    # configure and run the training/evaluation
    options = trw.train.Options(num_epochs=600)
    trainer = trw.train.TrainerV2(callbacks_post_training=None)

    transforms = [
        trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 16)),
        trw.transforms.TransformRandomCropPad(padding=[0, 4, 4]),
    ]

    #transforms = None

    model = Net_simple(options)
    results = trainer.fit(
        options,
        datasets=trw.datasets.create_cifar10_dataset(
            transform_train=transforms, nb_workers=2, batch_size=1000, data_processing_batch_size=500),
        log_path='cifar10_darts_search',
        #model_fn=lambda options: Net_DARTS(options),
        model=model,
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_fn(
            datasets=datasets, model=model, learning_rate=0.01))
    
    model.export_configuration(options.workflow_options.current_logging_directory)

    print('DONE')
