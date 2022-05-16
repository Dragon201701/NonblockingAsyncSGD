from enum import Enum
import torch
import torch.distributed as dist

from serialization import serialize_parameters, deserialize_parameters

class MPI_MessageCode(Enum):
    UpdateGradients = 0
    UpdateParameters = 1
    RequestParameters = 2
    EvaluateParameters = 3
    CloseServer = 4


class MPIListener():
    def __init__(self, model):
        self.model = model
        self.parameter = torch.zeros(serialize_parameters(model).numel() + 2)

    def receive_message(self, src, message_code, parameter):

        raise NotImplementedError()

    def run(self):
        self.running = True
        while self.running:
            dist.recv(tensor=self.parameter)
            self.receive_message(int(self.parameter[0].item()),
                                 MPI_MessageCode(self.parameter[1].item()),
                                 self.parameter[2:])


def send_message(message_code, payload, dst=0):
    parameters = torch.Tensor([dist.get_rank(), message_code.value])
    parameters = torch.cat((parameters, payload))
    return dist.isend(tensor=parameters, dst=dst)

def broadcast_message(model, message_code, src=0, group=None):
    rank = dist.get_rank()
    payload = serialize_parameters(model)
    parameters = torch.Tensor([rank,message_code.value])
    parameters = torch.cat((parameters, payload))
    if group is not None:
        dist.broadcast(tensor=parameters, src=src, group=group)
    else:
        dist.broadcast(tensor=parameters, src=src)

    if rank != src:
        deserialize_parameters(model, parameters[2:])

def receive_message(model, src=0):
    parameters = torch.zeros(serialize_parameters(model).numel() + 2)
    dist.recv(tensor=parameters, src=src)
    sender = int(parameters[0].item())
    message_code = MPI_MessageCode(parameters[1].item())
    if message_code == MPI_MessageCode.UpdateParameters:
        deserialize_parameters(model, parameters[2:])
