import torch
import numpy as np
import torch.nn as nn

nu = 0.01


class NavierStokes(nn.Module):
    def __init__(self, X, Y, T, u, v, device):
        super(NavierStokes, self).__init__()
        self.training_losses = []
        self.x = torch.tensor(X, dtype=torch.float32,
                              requires_grad=True).to(device)
        self.y = torch.tensor(Y, dtype=torch.float32,
                              requires_grad=True).to(device)
        self.t = torch.tensor(T, dtype=torch.float32,
                              requires_grad=True).to(device)

        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)

        # null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1)).to(device)

        # initialize network:

        self.net = self.network().to(device)

        self.lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)).to(device)
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)).to(device)

        self.optimizer = torch.optim.LBFGS(self.parameters(), lr=1, max_iter=50_000, max_eval=50_000,
                                           history_size=50, tolerance_grad=1e-05, tolerance_change=1.0 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")

        self.mse = nn.MSELoss().to(device)

        # loss
        self.ls = 0

        # iteration number
        self.iter = 0

    def network(self):

        return nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2))

    def function(self, x, y, t):

        res = self.net(torch.hstack((x, y, t)))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(
            psi), create_graph=True)[0]  # retain_graph=True,
        v = -1. * \
            torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(
                psi), create_graph=True)[0]

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        v_x = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(
            v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_y = torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_yy = torch.autograd.grad(
            v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_t = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        p_x = torch.autograd.grad(
            p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(
            p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = u_t + self.lambda_1*(u*u_x + v*u_y) + \
            p_x - self.lambda_2*(u_xx + u_yy)
        g = v_t + self.lambda_1*(u*v_x + v*v_y) + \
            p_y - self.lambda_2*(v_xx + v_yy)

        return u, v, p, f, g

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u, v, p, g and f predictions:
        u_prediction, v_prediction, p_prediction, f_prediction, g_prediction = self.function(
            self.x, self.y, self.t)

        # calculate losses
        u_loss = self.mse(u_prediction, self.u)
        v_loss = self.mse(v_prediction, self.v)
        f_loss = self.mse(f_prediction, self.null)
        g_loss = self.mse(g_prediction, self.null)
        self.ls = u_loss + v_loss + f_loss + g_loss

        # derivative with respect to net's weights:
        self.ls.backward()

        self.training_losses.append(self.ls.item())

        self.iter += 1
        if not self.iter % 1:
            print('Iteration: {:}, Loss: {:0.6f}, lambda_1: {:0.6f}, lambda_2: {:0.6f}'.format(
                self.iter, self.ls, float(self.lambda_1), float(self.lambda_2)))

        return self.ls

    def train(self):

        # training loop
        self.net.train()
        self.optimizer.step(self.closure)
        return self.training_losses, self.lambda_1.detach().numpy()[0], self.lambda_2.detach().numpy()[0]


# class PINN(nn.Module):
#     def __init__(self):
#         super(PINN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(3, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 20), nn.Tanh(),
#             nn.Linear(20, 2))

#     def forward(self, xyt):
#         return self.net(xyt)
