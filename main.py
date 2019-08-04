from torch import distributed
from torch import optim
from torch import utils
from torch import cuda
from torch import backends
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision import ops
from models import *
from distributed import *
from training import *
from utils import *
import numpy as np
import collections
import functools
import itertools
import argparse
import shutil
import json
import os


def main(args):

    init_process_group(backend='nccl')

    with open(args.config) as file:
        config = json.load(file)
    config.update(vars(args))
    config = apply_dict(Dict, config)

    backends.cudnn.benchmark = True
    backends.cudnn.fastest = True

    world_size = distributed.get_world_size()
    global_rank = distributed.get_rank()
    device_count = cuda.device_count()
    local_rank = global_rank % device_count

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cuda.manual_seed(config.seed)
    cuda.set_device(local_rank)

    train_dataset = datasets.MNIST(
        root=config.train_root,
        train=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True
    )
    val_dataset = datasets.MNIST(
        root=config.val_root,
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True
    )

    train_sampler = utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_data_loader = utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    generator = Generator(
        latent_size=128,
        mapping_layers=2,
        min_resolution=4,
        max_resolution=32,
        max_channels=128,
        min_channels=16,
        out_channels=1
    ).cuda()

    discriminator = Discriminator(
        in_channels=1,
        min_channels=16,
        max_channels=128,
        max_resolution=32,
        min_resolution=4,
        num_classes=1
    ).cuda()

    inverter = Discriminator(
        in_channels=1,
        min_channels=16,
        max_channels=128,
        max_resolution=32,
        min_resolution=4,
        num_classes=128
    ).cuda()

    config.global_batch_size = config.local_batch_size * distributed.get_world_size()
    config.generator_optimizer.lr *= config.global_batch_size / config.global_batch_denom
    config.discriminator_optimizer.lr *= config.global_batch_size / config.global_batch_denom
    config.inverter_optimizer.lr *= config.global_batch_size / config.global_batch_denom

    generator_optimizer = optim.Adam(generator.parameters(), **config.generator_optimizer)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), **config.discriminator_optimizer)
    inverter_optimizer = optim.Adam(inverter.parameters(), **config.inverter_optimizer)

    trainer = GANTrainer(
        latent_size=128,
        generator=generator,
        discriminator=discriminator,
        inverter=inverter,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        inverter_optimizer=inverter_optimizer,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        log_dir=os.path.join('log', config.name)
    )

    if config.checkpoint:
        trainer.load(config.checkpoint)

    if config.training:
        for epoch in range(trainer.epoch, config.num_epochs):
            trainer.step(epoch)
            trainer.train()
            # trainer.validate()
            trainer.save()

    elif config.validation:
        trainer.validate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generating Natural Adversarial Examples')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='train_gan')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
