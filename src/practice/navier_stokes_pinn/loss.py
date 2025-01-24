import torch
from torch.nn import MSELoss

nu = 0.01
mse = MSELoss()


def closure(optimizer, output, x, y, t, u, v):
    optimizer.zero_grad()
    u_hat, v_hat, _, f_hat, g_hat = post_nn(output, x, y, t)
    zeros = torch.zeros((x.shape[0], 1))
    u_loss = mse(u_hat, u)
    v_loss = mse(v_hat, v)
    f_loss = mse(f_hat, zeros)
    g_loss = mse(g_hat, zeros)
    final_loss = u_loss + v_loss + f_loss + g_loss
    final_loss.backward()
    return final_loss


def calc_loss(output, x, y, t, u, v):
    u_hat, v_hat, _, f_hat, g_hat = post_nn(output, x, y, t)
    zeros = torch.zeros((x.shape[0], 1))
    u_loss = mse(u_hat, u)
    v_loss = mse(v_hat, v)
    f_loss = mse(f_hat, zeros)
    g_loss = mse(g_hat, zeros)
    final_loss = u_loss + v_loss + f_loss + g_loss
    return final_loss


def post_nn(output, x, y, t):
    psi, p = output[:, 0:1], output[:, 1:2]
    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(
        psi), create_graph=True)[0]
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

    f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return u, v, p, f, g
