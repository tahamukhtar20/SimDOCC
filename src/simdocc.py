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

def compute_distance_metrics(embeddings, center):
    """
    Compute comprehensive distance statistics for clustering evaluation
    """
    distances = torch.sum((embeddings - center) ** 2, dim=1)
    return {
        'avg_distance': torch.mean(distances).item(),
        'std_distance': torch.std(distances).item(),
        'min_distance': torch.min(distances).item(),
        'max_distance': torch.max(distances).item(),
        'median_distance': torch.median(distances).item(),
        'distance_range': (torch.max(distances) - torch.min(distances)).item()
    }

def compute_per_class_distances(embeddings, labels, center):
    """
    Compute average distance per class - shows if all classes cluster equally well
    """
    class_distances = {}
    unique_classes = torch.unique(labels)
    
    for class_id in unique_classes:
        class_mask = labels == class_id
        class_embeddings = embeddings[class_mask]
        distances = torch.sum((class_embeddings - center) ** 2, dim=1)
        class_distances[class_id.item()] = torch.mean(distances).item()
    
    return class_distances

def log_distance_metrics(writer, model, dataloader, svdd_center, global_step, max_samples=2000):
    """
    Log comprehensive distance metrics instead of t-SNE
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        samples_collected = 0
        for inputs, targets in dataloader:
            if samples_collected >= max_samples:
                break
                
            inputs = inputs.cuda()
            _, features = model(inputs)
            
            embeddings_list.append(features.cpu())
            labels_list.append(targets.cpu())
            samples_collected += len(inputs)
    
    all_embeddings = torch.cat(embeddings_list, dim=0)[:max_samples]
    all_labels = torch.cat(labels_list, dim=0)[:max_samples]
    
    distance_stats = compute_distance_metrics(all_embeddings, svdd_center.cpu())
    
    writer.add_scalar('distance_analysis/avg_distance', distance_stats['avg_distance'], global_step)
    writer.add_scalar('distance_analysis/std_distance', distance_stats['std_distance'], global_step)
    writer.add_scalar('distance_analysis/distance_range', distance_stats['distance_range'], global_step)
    writer.add_scalar('distance_analysis/median_distance', distance_stats['median_distance'], global_step)
    
    class_distances = compute_per_class_distances(all_embeddings, all_labels, svdd_center.cpu())
    for class_id, avg_dist in class_distances.items():
        writer.add_scalar(f'distance_per_class/class_{class_id}', avg_dist, global_step)
    
    clustering_quality = distance_stats['std_distance'] / max(distance_stats['avg_distance'], 1e-8)
    writer.add_scalar('distance_analysis/clustering_quality', clustering_quality, global_step)
    
    print(f"Distance Analysis - Avg: {distance_stats['avg_distance']:.6f}, "
          f"Std: {distance_stats['std_distance']:.6f}, "
          f"Range: {distance_stats['distance_range']:.6f}, "
          f"Quality: {clustering_quality:.6f}")
    
    model.train()

def deep_svdd_loss(embeddings, center, scale_factor=1.0, radius_penalty=1.5):
    """
    One-Class Deep SVDD loss: minimize mean distance to center
    Args:
        embeddings: normalized SVDD features from model
        center: fixed hypersphere center
    """
    distances = torch.sum((embeddings - center) ** 2, dim=1)
    scaled_distances = distances * scale_factor
    squared_distances = scaled_distances ** 2
    return torch.mean(squared_distances) * radius_penalty

def initialize_svdd_center(model, train_loader, device='cuda'):
    """
    Initialize SVDD center as mean of initial forward pass
    This prevents hypersphere collapse as mentioned in the paper
    """
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= 10:
                break
            inputs = inputs.to(device)
            _, svdd_features = model(inputs)
            embeddings_list.append(svdd_features)
    
    all_embeddings = torch.cat(embeddings_list, dim=0)
    center = torch.mean(all_embeddings, dim=0)
    
    model.train()
    return center

def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, total_steps, writer=None, svdd_center=None, lambda_svdd=0.1):
    """Train for one epoch on the training set"""
    global global_step
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    svdd_losses = AverageMeter()
    distance_meters = AverageMeter()
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
            outputs, svdd_features = model(inputs)
            classification_loss = criterion(outputs, targets)

            if svdd_center is not None:
                svdd_loss = deep_svdd_loss(svdd_features, svdd_center)
                batch_distances = torch.sum((svdd_features - svdd_center) ** 2, dim=1)
                avg_batch_distance = torch.mean(batch_distances).item()
                total_loss = classification_loss + lambda_svdd * svdd_loss
            else:
                svdd_loss = torch.tensor(0.0)
                avg_batch_distance = 0.0
                total_loss = classification_loss

        prec1 = accuracy(outputs.data, targets, topk=(1,))[0]
        losses.update(total_loss.data.item(), inputs.size(0))
        classification_losses.update(classification_loss.data.item(), inputs.size(0))
        svdd_losses.update(svdd_loss.data.item(), inputs.size(0))
        distance_meters.update(avg_batch_distance, inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix({
            "Step": f"{global_step}/{total_steps}",
            "Loss": f"{losses.val:.4f} ({losses.avg:.4f})",
            "SVDD": f"{svdd_losses.val:.6f}",
            "Dist": f"{distance_meters.val:.6f}",
            "Prec@1": f"{top1.val:.3f} ({top1.avg:.3f})"
        })
        
        if global_step >= total_steps:
            print("Reached maximum number of steps, stopping training.")
            break

    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, global_step)
        writer.add_scalar('train/accuracy', top1.avg, global_step)
        writer.add_scalar('train/classification_loss', classification_losses.avg, global_step)
        writer.add_scalar('train/svdd_loss', svdd_losses.avg, global_step)

def validate(val_loader, model, criterion, epoch, writer=None, svdd_center=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    distance_meters = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}", leave=False)

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(pbar):
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)

            outputs, svdd_features = model(inputs)

            loss = criterion(outputs, targets)
            
            if svdd_center is not None:
                batch_distances = torch.sum((svdd_features - svdd_center) ** 2, dim=1)
                avg_batch_distance = torch.mean(batch_distances).item()
            else:
                avg_batch_distance = 0.0

            prec1 = accuracy(outputs.data, targets, topk=(1,))[0]
            losses.update(loss.data.item(), inputs.size(0))
            distance_meters.update(avg_batch_distance, inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({
                "Time": f"{batch_time.val:.3f} ({batch_time.avg:.3f})",
                "Loss": f"{losses.val:.4f} ({losses.avg:.4f})",
                "Dist": f"{distance_meters.val:.6f}",
                "Prec@1": f"{top1.val:.3f} ({top1.avg:.3f})"
            })

    print('[+] * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, global_step)
        writer.add_scalar('val/accuracy', top1.avg, global_step)
        writer.add_scalar('val/avg_distance_to_center', distance_meters.avg, global_step)
        
        if svdd_center is not None:
            log_distance_metrics(writer, model, val_loader, svdd_center, global_step, max_samples=1000)    
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

    svdd_center = initialize_svdd_center(model, train_loader)
    svdd_center = svdd_center.cuda()

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

        train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, total_steps, writer, svdd_center, lambda_svdd=0.8)

        prec1 = validate(val_loader, model, criterion, epoch, writer=writer, svdd_center=svdd_center)

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
