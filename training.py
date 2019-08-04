import torch
from torch import nn
from torch import distributed
from torch import autograd
from torchvision import models
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
from PIL import Image
from utils import *
import metrics
import os


class GANTrainer(object):

    def __init__(
        self,
        latent_size,
        generator,
        discriminator,
        inverter,
        generator_optimizer,
        discriminator_optimizer,
        inverter_optimizer,
        train_data_loader,
        val_data_loader,
        generator_lr_scheduler=None,
        discriminator_lr_scheduler=None,
        inverter_lr_scheduler=None,
        train_sampler=None,
        val_sampler=None,
        divergence_loss_weight=0.1,
        real_gradient_penalty_weight=0.0,
        fake_gradient_penalty_weight=0.0,
        log_steps=100,
        log_dir='log'
    ):

        self.latent_size = latent_size
        self.generator = generator
        self.discriminator = discriminator
        self.inverter = inverter
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.inverter_optimizer = inverter_optimizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.divergence_loss_weight = divergence_loss_weight
        self.real_gradient_penalty_weight = real_gradient_penalty_weight
        self.fake_gradient_penalty_weight = fake_gradient_penalty_weight
        self.log_steps = log_steps
        self.summary_dir = os.path.join(log_dir, 'summaries')
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.epoch = 0
        self.global_step = 0

        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.summary_writer = None
        if not self.rank:
            self.summary_writer = SummaryWriter(self.summary_dir)

        for tensor in self.generator.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)
        for tensor in self.discriminator.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)
        for tensor in self.inverter.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)

        # NOTE: Without doing this, all gradients is initialized to None.
        # NOTE: This causes that some of gradients of the same parameters on different devices can be None and cannot be reduced
        # NOTE: if they don't contribute to the loss because of path sampling.
        for parameter in self.generator.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        for parameter in self.discriminator.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        for parameter in self.inverter.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)

    def train(self):

        self.generator.train()
        self.discriminator.train()
        self.inverter.train()

        for real_images, _ in self.train_data_loader:

            batch_size = real_images.size(0)
            real_images = real_images.cuda(non_blocking=True)

            self.discriminator_optimizer.zero_grad()

            real_images.requires_grad_(True)
            real_logits = self.discriminator(real_images)

            fake_latents = torch.randn(batch_size, self.latent_size, 1, 1)
            fake_latents = fake_latents.cuda(non_blocking=True)

            with torch.no_grad():
                fake_images = self.generator(fake_latents)

            fake_images.requires_grad_(True)
            fake_logits = self.discriminator(fake_images)

            real_loss = torch.mean(nn.functional.softplus(-real_logits))
            fake_loss = torch.mean(nn.functional.softplus(fake_logits))
            discriminator_loss = real_loss + fake_loss

            if self.real_gradient_penalty_weight:
                real_gradients = autograd.grad(
                    outputs=real_logits,
                    inputs=real_images,
                    grad_outputs=torch.ones_like(real_logits),
                    retain_graph=True,
                    create_graph=True
                )[0]
                real_gradient_penalty = torch.mean(torch.sum(real_gradients ** 2, dim=(1, 2, 3)))
                discriminator_loss += real_gradient_penalty * self.real_gradient_penalty_weight

            if self.fake_gradient_penalty_weight:
                fake_gradients = autograd.grad(
                    outputs=fake_logits,
                    inputs=fake_images,
                    grad_outputs=torch.ones_like(fake_logits),
                    retain_graph=True,
                    create_graph=True
                )[0]
                fake_gradient_penalty = torch.mean(torch.sum(fake_gradients ** 2, dim=(1, 2, 3)))
                discriminator_loss += fake_gradient_penalty * self.fake_gradient_penalty_weight

            discriminator_loss.backward()

            for parameter in self.discriminator.parameters():
                if parameter.requires_grad:
                    distributed.all_reduce(parameter.grad)
                    parameter.grad /= self.world_size

            self.discriminator_optimizer.step()

            self.inverter_optimizer.zero_grad()

            fake_latents = torch.randn(batch_size, self.latent_size, 1, 1)
            fake_latents = fake_latents.cuda(non_blocking=True)

            with torch.no_grad():
                fake_images = self.generator(fake_latents)

            reconst_fake_latents = self.inverter(fake_images)
            reconst_real_latents = self.inverter(real_images)

            with torch.no_grad():
                reconst_real_images = self.generator(reconst_real_latents)

            divergence_loss = torch.mean((fake_latents - reconst_fake_latents) ** 2)
            reconstruction_loss = torch.mean((real_images - reconst_real_images) ** 2)
            inverter_loss = reconstruction_loss + divergence_loss * self.divergence_loss_weight

            inverter_loss.backward()

            for parameter in self.inverter.parameters():
                if parameter.requires_grad:
                    distributed.all_reduce(parameter.grad)
                    parameter.grad /= self.world_size

            self.inverter_optimizer.step()

            self.generator_optimizer.zero_grad()

            fake_latents = torch.randn(batch_size, self.latent_size, 1, 1)
            fake_latents = fake_latents.cuda(non_blocking=True)
            fake_images = self.generator(fake_latents)

            fake_logits = self.discriminator(fake_images)

            fake_loss = torch.mean(nn.functional.softplus(-fake_logits))
            generator_loss = fake_loss

            generator_loss.backward()

            for parameter in self.generator.parameters():
                if parameter.requires_grad:
                    distributed.all_reduce(parameter.grad)
                    parameter.grad /= self.world_size

            self.generator_optimizer.step()

            distributed.all_reduce(generator_loss)
            generator_loss /= self.world_size

            distributed.all_reduce(discriminator_loss)
            discriminator_loss /= self.world_size

            if not self.global_step % self.log_steps:
                self.log_scalars({
                    'generator_loss': generator_loss,
                    'discriminator_loss': discriminator_loss,
                    'inverter_loss': inverter_loss
                }, 'training')
                self.log_images(real_images, 'real_images')
                self.log_images(fake_images, 'fake_images')
                self.log_images(reconst_real_images, 'reconst_real_images')

            self.global_step += 1

    @torch.no_grad()
    def validate(self):

        def create_activation_generator(data_loader):

            def activation_generator():

                self.inception.eval()

                for real_images, _ in data_loader:

                    batch_size = real_images.size(0)
                    real_images = real_images.cuda(non_blocking=True)

                    latents = torch.randn(batch_size, self.latent_size, 1, 1)
                    latents = latents.cuda(non_blocking=True)
                    fake_images = self.generator(latents)

                    real_images = nn.functional.interpolate(real_images, size=(299, 299), mode="bilinear")
                    fake_images = nn.functional.interpolate(fake_images, size=(299, 299), mode="bilinear")

                    real_activations = self.inception(real_images)
                    fake_activations = self.inception(fake_images)

                    real_activations_list = [real_activations] * self.world_size
                    fake_activations_list = [fake_activations] * self.world_size

                    distributed.all_gather(real_activations_list, real_activations)
                    distributed.all_gather(fake_activations_list, fake_activations)

                    for real_activations, fake_activations in zip(real_activations_list, fake_activations_list):
                        yield real_activations, fake_activations

            return activation_generator

        self.generator.eval()
        self.discriminator.eval()

        real_activations, fake_activations = map(torch.cat, zip(*create_activation_generator(self.val_data_loader)()))
        frechet_inception_distance = metrics.frechet_inception_distance(real_activations.cpu().numpy(), fake_activations.cpu().numpy())
        self.log_scalars({'frechet_inception_distance': frechet_inception_distance}, 'validation')

    def log_scalars(self, scalars, tag):
        if self.summary_writer:
            for name, scalar in scalars.items():
                self.summary_writer.add_scalars(
                    main_tag=name,
                    tag_scalar_dict={tag: scalar},
                    global_step=self.global_step
                )
        if not self.rank:
            print(f'[{tag}] epoch: {self.epoch} global_step: {self.global_step} '
                  f'{" ".join([f"{key}: {value:.4f}" for key, value in scalars.items()])}')

    def log_images(self, images, tag):
        if self.summary_writer:
            self.summary_writer.add_image(
                tag=tag,
                img_tensor=vutils.make_grid(images, normalize=True),
                global_step=self.global_step
            )

    def save(self):
        if not self.rank:
            torch.save(dict(
                generator_state_dict=self.generator.state_dict(),
                discriminator_state_dict=self.discriminator.state_dict(),
                generator_optimizer_state_dict=self.generator_optimizer.state_dict(),
                discriminator_optimizer_state_dict=self.discriminator_optimizer.state_dict(),
                last_epoch=self.epoch,
                global_step=self.global_step
            ), os.path.join(self.checkpoint_dir, f'epoch_{self.epoch}'))

    def load(self, checkpoint):
        checkpoint = Dict(torch.load(checkpoint))
        self.generator.load_state_dict(checkpoint.generator_state_dict)
        self.discriminator.load_state_dict(checkpoint.discriminator_state_dict)
        self.generator_optimizer.load_state_dict(checkpoint.generator_optimizer_state_dict)
        self.discriminator_optimizer.load_state_dict(checkpoint.discriminator_optimizer_state_dict)
        self.epoch = checkpoint.last_epoch + 1
        self.global_step = checkpoint.global_step

    def step(self, epoch=None):
        self.epoch = self.epoch + 1 if epoch is None else epoch
        if self.generator_lr_scheduler:
            self.generator_lr_scheduler.step(self.epoch)
        if self.discriminator_lr_scheduler:
            self.discriminator_lr_scheduler.step(self.epoch)
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)
        if self.val_sampler:
            self.val_sampler.set_epoch(self.epoch)
