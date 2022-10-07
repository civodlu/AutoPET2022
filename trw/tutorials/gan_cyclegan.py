import collections
import itertools

import torch
import trw
from trw.layers.gan import process_outputs_and_extract_loss
from trw.train import get_device
from trw.transforms import TransformResize, TransformRandomFlip, TransformRandomCropPad
from torch import nn
from trw.utils import len_batch


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class CycleGAN:
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,
                 netG_A,
                 netG_B,
                 netD_A,
                 netD_B,
                 real_image_A_from_batch_fn,
                 real_image_B_from_batch_fn,
                 lambda_A: float = 10.0,
                 lambda_B: float = 10.0,
                 lambda_identity: float = 0.5,
                 fake_A_pool=None,
                 fake_B_pool=None
    ):

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity
        self.real_image_A_from_batch_fn = real_image_A_from_batch_fn
        self.real_image_B_from_batch_fn = real_image_B_from_batch_fn

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = netG_A
        self.netG_B = netG_B

        self.netD_A = netD_A
        self.netD_B = netD_B

        self.fake_A_pool = fake_A_pool
        self.fake_B_pool = fake_B_pool
        # define loss functions
        self.criterionGAN = GANLoss('lsgan')
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    def _set_input(self, batch):
        self.real_A = self.real_image_A_from_batch_fn(batch)
        self.real_B = self.real_image_B_from_batch_fn(batch)

    def forward(self, batch):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(batch, self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(batch, self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(batch, self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(batch, self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self, batch):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.get_data(batch, self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self, batch):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.get_data(batch, self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self, batch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._set_input(batch)

        # forward
        self.forward(batch)      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A(batch)      # calculate gradients for D_A
        self.backward_D_B(batch)      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    @staticmethod
    def set_requires_grad(nets, requires_grad):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


transforms = [
    TransformResize(size=[286, 286]),
    TransformRandomFlip(axis=3),
    TransformRandomCropPad(padding=None, shape=[3, 256, 256]),
]

datasets = trw.datasets.create_cycle_gan_dataset('facades', aligned=True, transform_train=transforms)
for d_name, d in datasets.items():
    for split_name, split in d.items():
        for batch in split:
            print('TEST')

print('DONE')
