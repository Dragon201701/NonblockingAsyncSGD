import torch
import torch.distributed as dist
from torch.optim import Adam

from messaging import receive_message, send_message, broadcast_message, MPI_MessageCode, MPIListener
from serialization import serialize_parameters, deserialize_parameters

class DownpourSGD():
    def __init__(self, model, n_step=2):
        rank = dist.get_rank()
        wsize = dist.get_world_size()
        
        if rank == 1:
            _p = send_message(MPI_MessageCode.UpdateParameters, torch.tensor([0.0]))
            _p.wait()
        
        broadcast_message(model, MPI_MessageCode.UpdateParameters, src=0)
        
        self.optimizer = Adam(model.parameters())
        self.lr = self.optimizer.defaults['lr']
        self.accumulated_gradients = torch.zeros(
            serialize_parameters(model).size())
        
        self.model = model
        self.n_step = n_step
        self.idx = 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        gradients = serialize_parameters(self.model, grad=True)
        self.accumulated_gradients.add_(gradients)
        self.optimizer.step()

        if self.idx % self.n_step == 0:
            _p = send_message(MPI_MessageCode.UpdateGradients,
                         self.accumulated_gradients)
            _p.wait()
            _p = send_message(MPI_MessageCode.RequestParameters, torch.Tensor())
            _p.wait()
            receive_message(self.model)
            self.accumulated_gradients.zero_()
        
        self.idx += 1
        

