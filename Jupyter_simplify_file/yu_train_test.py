import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
import os

import numpy as np

from models.losses.p2m import P2MLoss
from utils.mesh import Ellipsoid
from utils.average_meter import AverageMeter
from functions.evaluator import Evaluator

def summarize_model(model):
    layers = [(name if len(name) > 0 else 'TOTAL', str(module.__class__.__name__), sum(np.prod(p.shape) for p in module.parameters())) for name, module in model.named_modules()]
    layers.append(layers[0])
    del layers[0]

    columns = [
        [" ", list(map(str, range(len(layers))))],
        ["Name", [layer[0] for layer in layers]],
        ["Type", [layer[1] for layer in layers]],
        ["Params", [layer[2] for layer in layers]],
    ]

    n_rows = len(columns[0][1])
    n_cols = 1 + len(columns)

    # Get formatting width of each column
    col_widths = []
    for c in columns:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(columns, col_widths)]

    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(columns, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += "\n" + " | ".join(line)

    return summary

def yu_train(options, model, ellipsoid, device, trainloader, num_epochs):

    # declare optimizer
    optimizer = torch.optim.Adam(
                params=list(model.parameters()),
                lr=options.optim.lr,
                betas=(options.optim.adam_beta1, 0.999),
                weight_decay=options.optim.wd
            )
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, options.optim.lr_step, options.optim.lr_factor
        )
    
     # loss
    
    loss_criterion = P2MLoss(options.loss, ellipsoid).to(device)

    
    # for loss summary
    losses = AverageMeter()

     # evalutaor
    evaluators = None

    model.to(device)

    epoch_count = 0

    for epoch in range(num_epochs):
        epoch_count += 1
      
        losses.reset()
        for i, batch in enumerate(trainloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            images = batch["images"]
            
            model.train()
            
            out = model(images)

            loss, loss_summary = loss_criterion(out, batch)

            # print("loss")
            # print(loss.detach().item())

            iteration = epoch * len(trainloader) + i

            # if iteration % 5 == 4:
            #     print(f'[{epoch:03d}/{i:05d}] train_loss: {loss.detach().item():.3f}')

                

            losses.update(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        print('epoch:{}'.format(epoch_count))
        print('current_loss (average_loss): {}'.format(losses))

    # model.to_dense()
    # torch.save(model().state_dict(), os.path.join(config.YU_CHECKPOINT_PATH, '/yu_model.ckpt'))


