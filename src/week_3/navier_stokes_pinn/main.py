import os
import model
import utils
import numpy as np
from datetime import datetime
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    x_train, y_train, t_train, u_train, v_train, x_test, y_test, t_test, u_test, v_test, p_test = utils.get_data(
        './data/cylinder_nektar_wake.mat', 5000)

    pinn = model.NavierStokes(
        x_train, y_train, t_train, u_train, v_train, device)

    output_path = os.path.join('./output', datetime.strftime(
        datetime.now(), '%Y-%m-%d %H:%M:%S'))

    os.mkdir(output_path)

    losses, lambda_1, lambda_2 = pinn.train()

    np.save(os.path.join(output_path, 'training_losses.npy'), np.array(losses))
    np.save(os.path.join(output_path, 'lambdas.npy'), np.array([lambda_1, lambda_2]))
    torch.save(pinn.net.state_dict(), os.path.join(output_path, 'model.pt'))
    # x_train_tensor = torch.tensor(
    #     x_train, dtype=torch.float32, device=device, requires_grad=True)
    # y_train_tensor = torch.tensor(
    #     y_train, dtype=torch.float32, device=device, requires_grad=True)
    # t_train_tensor = torch.tensor(
    #     t_train, dtype=torch.float32, device=device, requires_grad=True)
    # u_train_tensor = torch.tensor(u_train,  dtype=torch.float32, device=device)
    # v_train_tensor = torch.tensor(v_train,  dtype=torch.float32, device=device)

    # x_test_tensor = torch.tensor(
    #     x_test, dtype=torch.float32, device=device, requires_grad=True)
    # y_test_tensor = torch.tensor(
    #     y_test, dtype=torch.float32, device=device, requires_grad=True)
    # t_test_tensor = torch.tensor(
    #     t_test, dtype=torch.float32, device=device, requires_grad=True)
    # u_test_tensor = torch.tensor(u_test,  dtype=torch.float32, device=device)
    # v_test_tensor = torch.tensor(v_test, dtype=torch.float32, device=device)

    # pinn = model.PINN().to(device)
    # optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1, max_iter=50_000, max_eval=50_000,
    #                               history_size=50, tolerance_grad=1e-05, tolerance_change=1.0 * np.finfo(float).eps,
    #                               line_search_fn="strong_wolfe")
    # output_path = os.path.join('./output', datetime.strftime(
    #     datetime.now(), '%Y-%m-%d %H:%M:%S'))
    # os.mkdir(output_path)
    # trainer.train(pinn, 200_000, optimizer, output_path, x_train_tensor, y_train_tensor, t_train_tensor,
    #               u_train_tensor, v_train_tensor, x_test_tensor, y_test_tensor, t_test_tensor, u_test_tensor, v_test_tensor)


if __name__ == '__main__':
    main()
