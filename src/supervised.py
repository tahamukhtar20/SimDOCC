import argparse
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import WideResNet
from dataloaders import create_dataloaders
from utils import (
    AverageMeter,
    accuracy,
    save_checkpoint,
    get_cosine_schedule_with_warmup
)

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=1400, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor (default: 2)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-2', type=str, help='name of experiment')

parser.set_defaults(augment=True)

best_prec1 = 0
global_step = 0

def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, total_steps, writer=None):
    """Train for one epoch on the training set"""
    global global_step
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    end = time.time()
    for i, (inputs, targets) in enumerate(pbar):
        if global_step >= total_steps:
            print("Reached maximum number of steps, stopping training.")
            break
        
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        prec1 = accuracy(outputs.data, targets, topk=(1,))[0]
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix({
            "Step": f"{global_step}/{total_steps}",
            "Time": f"{batch_time.val:.3f} ({batch_time.avg:.3f})",
            "Loss": f"{losses.val:.4f} ({losses.avg:.4f})",
            "Prec@1": f"{top1.val:.3f} ({top1.avg:.3f})"
        })
        
        if global_step >= total_steps:
            print("Reached maximum number of steps, stopping training.")
            break

    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, global_step)
        writer.add_scalar('train/accuracy', top1.avg, global_step)

def validate(val_loader, model, criterion, epoch, writer=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}", leave=False)

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(pbar):
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            prec1 = accuracy(outputs.data, targets, topk=(1,))[0]
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({
                "Time": f"{batch_time.val:.3f} ({batch_time.avg:.3f})",
                "Loss": f"{losses.val:.4f} ({losses.avg:.4f})",
                "Prec@1": f"{top1.val:.3f} ({top1.avg:.3f})"
            })

    print('[+] * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, global_step)
        writer.add_scalar('val/accuracy', top1.avg, global_step)
    return top1.avg

def main():
    global args, best_prec1, global_step
    args = parser.parse_args()
    
    writer = SummaryWriter()
    
    scaler = GradScaler('cuda')

    train_loader, val_loader = create_dataloaders(args.dataset, args.augment, args.batch_size)

    model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            global_step = checkpoint.get('global_step', 0)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, step {})"
                  .format(args.resume, checkpoint['epoch'], global_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    total_steps = 2**20
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps, num_warmup_steps=0)

    if global_step > 0:
        for _ in range(global_step):
            scheduler.step()

    print(f"Training for {total_steps} total steps")
    print(f"Starting from step {global_step}")

    for epoch in range(args.start_epoch, args.epochs):
        if global_step >= total_steps:
            print(f"Reached maximum steps ({total_steps}). Stopping training.")
            break

        train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, total_steps, writer)

        prec1 = validate(val_loader, model, criterion, epoch, writer=writer)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'global_step': global_step,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args.name)

        if global_step >= total_steps:
            break
            
    if writer is not None:
        writer.close()

    print('Best accuracy: ', best_prec1)
    print('Final step: ', global_step)

if __name__ == '__main__':
    main()
