import torch
import math
from torch.autograd import Variable

def angle_normalize(x):
       return (((x + math.pi) % (2 * math.pi)) - math.pi)
 
class PendulumDynamics(torch.nn.Module):
       def forward(self, state, action):
              th = state[:, 0].view(-1, 1)
              thdot = state[:, 1].view(-1, 1)
              # print(f'th {th}, thdot {thdot}')

              g = 10
              m = 1.0
              l = 1.0
              dt = 0.05

              u = action
              u = torch.clamp(u, -2, 2)

              newthdot = thdot + (3 * g / (2 * l) * torch.sin(th ) + 3. / (m * l ** 2) * u) * dt
              newth = th + newthdot * dt
              newthdot = torch.clamp(newthdot, -8, 8)

              state = torch.cat((angle_normalize(newth), newthdot), dim=1)
              # print(f'state {state}')
              return state
       
       def grad_input(self, state, action):
              assert isinstance(state, Variable) == isinstance(action, Variable)
              # print(f'state {state}, action {action}')
              th = state[:, 0]
              # print(f'th {th}')

              x_dim, u_dim = state.ndimension(), action.ndimension()
              # print(f'x_dim {x_dim}, u_dim {u_dim}')
              n_batch, n_state = state.size()
              _, n_ctrl = action.size()

              g = 10
              m = 1
              l = 1
              

              A = torch.zeros((n_batch, n_state, n_state)).to(state.device)
              B = torch.zeros((n_batch, n_state, n_ctrl)).to(state.device)
              A[:,0, 1] = 1
              A[:,1, 0] = 3 * g / (2 * l) * torch.cos(th)
              B[:,1, 0] = 3. / (m * l ** 2)

              return A, B
              