import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from torch.nn import init

import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator2(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator2(nn.Module):
    def __init__(self, input_nc):
        super().__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Generator(nn.Module):
    def __init__(self, latent_size):
        latent_size = 0
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.unet = trw.layers_legacy.UNetBase(2, input_channels=3 + latent_size, output_channels=3, channels=[64, 128, 256, 512])

    def forward(self, segmentation):
        #latent = latent.repeat([1, 1, segmentation.shape[2], segmentation.shape[3]])
        #i = torch.cat([latent, segmentation], dim=1)
        #x = self.unet(i)
        x = self.unet(segmentation)
        return torch.tanh(x)  # force -1..1 range


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convs = trw.layers_legacy.convs_2d(
            3,
            [64, 128, 256, 512, 512, 1],
            convolution_kernels=[4, 4, 4, 4, 4, 1],
            strides=[2, 2, 2, 2, 1, 1],
            batch_norm_kwargs={},
            pooling_size=None,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True,
            bias=False
        )

    def forward(self, image):
        #x = self.convs(torch.cat([image, segmentation], dim=1))
        x = self.convs(image)
        return x


def get_facade(batch):
    return 2 * batch['images'] - 1


def get_mask(batch):
    return 2 * batch['segmentations'] - 1


class Object:
    pass


def create_model(options):
    discriminator_facades = Discriminator2(3)
    generator_mask_to_facades = Generator2(3, 3)

    discriminator_mask = Discriminator2(3)
    generator_facades_to_mask = Generator2(3, 3)

    opt = Object()
    opt.input_nc = 3
    opt.output_nc = 3
    opt.pool_size = 50
    opt.gan_mode = 'lsgan'
    opt.lr = 0.0002
    opt.beta1 = 0.5
    opt.lambda_identity = 0.5
    opt.lambda_A = 10
    opt.lambda_B = 10

    m = trw.layers_legacy.CycleGan(
        opt=opt,
        netG_A=generator_mask_to_facades,
        netG_B=generator_facades_to_mask,
        netD_A=discriminator_facades,
        netD_B=discriminator_mask,
        get_real_A_fn=get_mask,
        get_real_B_fn=get_facade
    )

    m.apply(init_weights)
    return m


def per_epoch_callbacks():
    return [
        trw.callbacks.CallbackSkipEpoch(1, [
            trw.callbacks.CallbackReportingExportSamples(split_exclusions=['valid', 'test']),
        ], include_epoch_zero=True),
        trw.callbacks.CallbackEpochSummary(),
    ]


def pre_training_callbacks():
    return [
        trw.callbacks.CallbackReportingStartServer(),
    ]


options = trw.train.Options(num_epochs=600, device='cuda:0')

trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
    callbacks_pre_training_fn=pre_training_callbacks)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=1,
    inputs_fn=lambda: trw.datasets.create_facades_dataset(
        batch_size=10,
        transforms_train=trw.transforms.TransformCompose([
            trw.transforms.TransformRandomFlip(axis=3),
            trw.transforms.TransformRandomCropPad([0, 16, 16], mode='symmetric')
        ])),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_cycle_gan',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')