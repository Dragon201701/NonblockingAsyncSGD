import torch.nn as nn
import torch
import torch.distributed as dist

from config import get_config
from data_loader import get_data_loader, partition_dataset
from model import Net
from optimizer import DownpourSGD
from parameter_server import ParameterServer
from messaging import MPI_MessageCode, send_message, broadcast_message
import time


"""
Author: Haoze He
NYU ID: hh2537
Email: hh2537@nyu.edu
JOB NAME: HPML-Final-Project
Project title: Non-blocking Synchronous SGD: a novel communication scheme for distributed deep learning
"""




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=3):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    _, predicted = output.topk(topk, 1, True, True)
    batch_size = target.size(0)
    prec_k = 0
    prec_1 = 0
    count_k = 0
    for i in range(batch_size):
        prec_k += target[i][predicted[i]].sum()
        prec_1 += target[i][predicted[i][0]]
        count_k += topk  # min(target[i].sum(), topk)
    prec_k /= count_k
    prec_1 /= batch_size
    return prec_1.item(), prec_k.item()


def param_server(config):
    device = torch.device(
        "cuda" if config.gpu and torch.cuda.is_available() else "cpu")
    model = Net(3, 17).to(device)
    server = ParameterServer(model)
    if config.log:
        print("Starting parameter server")

    server.run()

    if config.log:
        print("Stopping parameter server")


def train(config, worker_group):
    rank = dist.get_rank()
    wsize = dist.get_world_size()
    device = torch.device(
        "cuda" if config.gpu and torch.cuda.is_available() else "cpu")
    dataloader = partition_dataset(config)
    model = Net(3, 17).to(device)
    optimizer = DownpourSGD(model, n_step=config.n_step)
    loss_fn = nn.BCELoss()
    model.train()

    avg_loss = AverageMeter()
    avg_top1 = AverageMeter()
    avg_top3 = AverageMeter()
    total_time = 0.0

    for epoch in range(1, config.epochs+1):
        start_time = time.monotonic()
        total_sample = 0
        count = 0
        avg_loss.reset()
        avg_top1.reset()
        avg_top3.reset()

        for batch_idx, (data, target) in enumerate(dataloader, 1):
            total_sample += data.shape[0]
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            top1, top3 = accuracy(output, target)

            avg_loss.update(loss.item())
            avg_top1.update(top1)
            avg_top3.update(top3)

            if config.log and batch_idx % config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} precision@1: {:.6f} precision@3: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item(), top1, top3))

        dist.barrier(worker_group)

        if rank == 1:
            _p = send_message(MPI_MessageCode.UpdateParameters, torch.tensor([0.0]))
            _p.wait()
        
        broadcast_message(model, MPI_MessageCode.UpdateParameters, src=0)

        epoch_time = time.monotonic() - start_time
        print('Rank: {} Epoch: {} loss: {:.6f} prec@1: {:.6f} prec@3: {:.6f} exec_time: {}'.format(
            rank, epoch, avg_loss.avg, avg_top1.avg, avg_top3.avg, epoch_time))
        
        total_time += epoch_time

    loss_sum = torch.tensor(total_sample*avg_loss.avg)
    top1_sum = torch.tensor(total_sample*avg_top1.avg)
    top3_sum = torch.tensor(total_sample*avg_top3.avg)
    total_sample_sum = torch.tensor(total_sample)

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=worker_group)
    dist.all_reduce(top1_sum, op=dist.ReduceOp.SUM, group=worker_group)
    dist.all_reduce(top3_sum, op=dist.ReduceOp.SUM, group=worker_group)
    dist.all_reduce(total_sample_sum,
                    op=dist.ReduceOp.SUM, group=worker_group)

    avg_loss_value = loss_sum.item() / total_sample_sum.item()
    avg_top1_value = top1_sum.item() / total_sample_sum.item()
    avg_top3_value = top3_sum.item() / total_sample_sum.item()

    print('Rank: {} loss: {:.6f} prec@1: {:.6f} prec@3: {:.6f} total_exec_time: {}'.format(
        rank, avg_loss_value, avg_top1_value, avg_top3_value, total_time))


    if rank == 1:
        _p = send_message(MPI_MessageCode.CloseServer, torch.tensor([0.0]))
        _p.wait()


def main(config):
    rank = dist.get_rank()
    wsize = dist.get_world_size()
    worker_group = dist.new_group(range(1, wsize))
    if rank == 0:
        param_server(config)
    else:
        if config.log:
            print('Node: {}/{}: Starting worker'.format(rank, wsize))
        train(config, worker_group)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    dist.init_process_group(backend='mpi')
    main(get_config())



