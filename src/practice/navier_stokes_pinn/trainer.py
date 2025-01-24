import os
import torch
import numpy as np
from loss import closure


def train(model, n_epochs, optimizer, output_path, x_train, y_train, t_train, u_train, v_train, x_test, y_test, t_test, u_test, v_test):
    model_path = os.path.join(output_path, 'model')
    loss_path = os.path.join(output_path, 'losses')
    os.mkdir(model_path)
    os.mkdir(loss_path)
    xyt_train = torch.hstack((x_train, y_train, t_train))
    # xyt_test = torch.hstack((x_test, y_test, t_test))
    training_losses = []
    # test_losses = []
    for epoch in range(1, n_epochs+1):
        model.train()
        output = model(xyt_train)
        # train_ls = calc_loss(output, x_train, y_train, t_train, u_train, v_train)
        optimizer.zero_grad()
        # train_ls.backward()
        print(optimizer.step(closure(optimizer, output, x_train, y_train, t_train, u_train, v_train)))
        # training_losses.append(train_ls.item())
        
        # print('\nepoch: ', epoch, 'training loss: ', train_ls.item())

        if epoch % 1000 == 0:
            # model.eval()
            # output = model(xyt_test)
            # test_ls = calc_loss(output, x_test, y_test, t_test, u_test, v_test)
            # test_losses.append(test_ls.item())
            # print(' test loss: ', test_ls.item(), end='')
            torch.save(model.state_dict(), model_path + f'/pinn_navier_stokes_{epoch}.pt')
    torch.save(model.state_dict(), model_path + f'/pinn_navier_stokes_{epoch}_final.pt')
    np.save(np.array(training_losses), loss_path + '/train.npy')
    # np.save(np.array(test_losses), loss_path + '/test.npy')
    return model
    
