import torch
from torch.autograd import Variable

 
class MountainCarDynamics(torch.nn.Module):
       def forward(self, state, action):
              # states
              position = state[:, 0].view(-1, 1)
              velocity = state[:, 1].view(-1, 1)
        
              # constants
              power = 0.0015
              
              # action
              force = torch.clamp(action, -1, 1)

              # dynamics
              velocity += force * power - 0.0025 * torch.cos(3 * position)
              velocity = torch.clamp(velocity, -0.07, 0.07)
              position += velocity
              position = torch.clamp(position, -1.2, 0.6)
              if (position == -1.2 and velocity < 0): velocity = 0
              state = torch.cat((position, velocity), dim=1)
              return state
       
       def grad_input(self, state, action):
              assert isinstance(state, Variable) == isinstance(action, Variable)
              # th = state[:, 0]
              position = state[:, 0]

              x_dim, u_dim = state.ndimension(), action.ndimension()
              n_batch, n_state = state.size()
              _, n_ctrl = action.size()
              
              # constants
              power = 0.0015

              A = torch.zeros((n_batch, n_state, n_state)).to(state.device)
              B = torch.zeros((n_batch, n_state, n_ctrl)).to(state.device)
              
              A[:,0, 1] = 1
              A[:,1, 0] = 0.0025 * 3 * torch.sin(3 * position)
              B[:,1, 0] = power

              return A, B
              