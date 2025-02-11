# import torch
# import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ConvLSTMCell(nn.Module):

#     def __init__(
#         self, in_channels, out_channels, kernel_size, padding, activation, frame_size
#     ):

#         super(ConvLSTMCell, self).__init__()

#         if activation == "tanh":
#             self.activation = torch.tanh
#         elif activation == "relu":
#             self.activation = torch.relu

#         self.conv = nn.Conv2d(
#             in_channels=in_channels + out_channels,
#             out_channels=4 * out_channels,
#             kernel_size=kernel_size,
#             padding=padding,
#         )

#         self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

#     def forward(self, X, H_prev, C_prev):

#         conv_output = self.conv(torch.cat([X, H_prev], dim=1))

#         i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

#         input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
#         forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

#         C = forget_gate * C_prev + input_gate * self.activation(C_conv)

#         output_gate = torch.sigmoid(o_conv + self.W_co * C)

#         H = output_gate * self.activation(C)

#         return H, C


# class ConvLSTM(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         num_kernels,
#         out_channels,
#         kernel_size,
#         padding,
#         activation,
#         frame_size,
#     ):
#         super(ConvLSTM, self).__init__()

#         self.out_channels = out_channels
#         self.num_kernels = num_kernels

#         self.convLSTMcell = ConvLSTMCell(
#             in_channels, num_kernels, kernel_size, padding, activation, frame_size
#         )

#         # self.batchnorm = nn.BatchNorm3d(num_features=num_kernels)

#         self.conv = nn.Conv2d(
#             in_channels=num_kernels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             padding=padding,
#         )
#         self.activation = nn.Sigmoid()

#     def forward(self, x, seq_len=None, teacher_forcing=True):
#         assert (
#             teacher_forcing or seq_len is not None
#         ), f"seq_len cannot be None when teacher_forcing is False"
#         if teacher_forcing:
#             # Get the dimensions
#             batch_size, _, seq_len, height, width = x.size()

#             # Initialize output
#             output = torch.zeros(
#                 batch_size, self.out_channels, seq_len, height, width, device=device
#             )

#             # Initialize Hidden State
#             H = torch.zeros(batch_size, self.num_kernels, height, width, device=device)

#             # Initialize Cell Input
#             C = torch.zeros(batch_size, self.num_kernels, height, width, device=device)

#             # Unroll over time steps
#             for time_step in range(seq_len):
                
#                 H, C = self.convLSTMcell(x[:, :, time_step], H, C)
#                 yhat = self.conv(H)
#                 yhat = self.activation(yhat)
#                 output[:, :, time_step] = yhat

#             return output
#         else:
#             batch_size, _, height, width = x.size()
#             output = torch.zeros(
#                 batch_size, self.out_channels, seq_len, height, width, device=device
#             )

#             # Initialize Hidden State
#             H = torch.zeros(batch_size, self.num_kernels, height, width, device=device)

#             # Initialize Cell Input
#             C = torch.zeros(batch_size, self.num_kernels, height, width, device=device)

#             # Unroll over time steps
#             for time_step in range(seq_len):
#                 H, C = self.convLSTMcell(x, H, C)
#                 x = self.activation(self.conv(H))
#                 output[:, :, time_step] = x

#             return output
