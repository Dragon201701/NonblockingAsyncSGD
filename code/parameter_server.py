import torch
from torch.optim import Adam
import torch.distributed as dist
from messaging import MPIListener, send_message, receive_message, broadcast_message, MPI_MessageCode
from serialization import serialize_parameters, deserialize_parameters

class ParameterServer(MPIListener):
    def __init__(self, model):
        self.model = model
        self.optimizer = Adam(self.model.parameters())
        self.optimizer.zero_grad()
        
        # initial_parameters = serialize_parameters(model)
        # send_process = []
        # for dst in range(1, dist.get_world_size()):
        #     send_process.append(send_message(MPI_MessageCode.UpdateParameters, initial_parameters, dst))
        
        # for _p in send_process:
        #     _p.wait()

        self.model = model
        super().__init__(model)
    
    def receive_message(self, src, message_code, parameters):

        if message_code == MPI_MessageCode.UpdateParameters:
            broadcast_message(self.model, message_code, src=0)
        
        elif message_code == MPI_MessageCode.RequestParameters:
            _p = send_message(MPI_MessageCode.UpdateParameters, serialize_parameters(self.model), src)
            _p.wait()
        
        elif message_code == MPI_MessageCode.UpdateGradients:
            self.optimizer.zero_grad()
            deserialize_parameters(self.model, parameters, grad=True)
            self.optimizer.step()
        
        elif message_code == MPI_MessageCode.CloseServer:
            self.running = False
