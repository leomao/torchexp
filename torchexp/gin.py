import torch as th
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import gin


gin.external_configurable(DataLoader, module='torch.utils.data')


# Losses
gin.external_configurable(nn.L1Loss, module='th.nn')
gin.external_configurable(nn.MSELoss, module='th.nn')
gin.external_configurable(nn.CrossEntropyLoss, module='th.nn')
gin.external_configurable(nn.CTCLoss, module='th.nn')
gin.external_configurable(nn.NLLLoss, module='th.nn')
gin.external_configurable(nn.PoissonNLLLoss, module='th.nn')
gin.external_configurable(nn.KLDivLoss, module='th.nn')
gin.external_configurable(nn.BCELoss, module='th.nn')
gin.external_configurable(nn.BCEWithLogitsLoss, module='th.nn')
gin.external_configurable(nn.MarginRankingLoss, module='th.nn')
gin.external_configurable(nn.HingeEmbeddingLoss, module='th.nn')
gin.external_configurable(nn.MultiLabelMarginLoss, module='th.nn')
gin.external_configurable(nn.SmoothL1Loss, module='th.nn')
gin.external_configurable(nn.SoftMarginLoss, module='th.nn')
gin.external_configurable(nn.MultiLabelSoftMarginLoss, module='th.nn')
gin.external_configurable(nn.CosineEmbeddingLoss, module='th.nn')
gin.external_configurable(nn.MultiMarginLoss, module='th.nn')
gin.external_configurable(nn.TripletMarginLoss, module='th.nn')


# Activation functions
gin.external_configurable(nn.ELU, module='th.nn')
gin.external_configurable(nn.Hardshrink, module='th.nn')
gin.external_configurable(nn.Hardtanh, module='th.nn')
gin.external_configurable(nn.LeakyReLU, module='th.nn')
gin.external_configurable(nn.MultiheadAttention, module='th.nn')
gin.external_configurable(nn.PReLU, module='th.nn')
gin.external_configurable(nn.ReLU, module='th.nn')
gin.external_configurable(nn.ReLU6, module='th.nn')
gin.external_configurable(nn.RReLU, module='th.nn')
gin.external_configurable(nn.SELU, module='th.nn')
gin.external_configurable(nn.CELU, module='th.nn')
gin.external_configurable(nn.Sigmoid, module='th.nn')
gin.external_configurable(nn.Softplus, module='th.nn')
gin.external_configurable(nn.Softshrink, module='th.nn')
gin.external_configurable(nn.Softsign, module='th.nn')
gin.external_configurable(nn.Tanh, module='th.nn')
gin.external_configurable(nn.Tanhshrink, module='th.nn')
gin.external_configurable(nn.Threshold, module='th.nn')


# Optimizers
gin.external_configurable(optim.Adadelta, module='th.optim')
gin.external_configurable(optim.Adagrad, module='th.optim')
gin.external_configurable(optim.Adam, module='th.optim')
gin.external_configurable(optim.SparseAdam, module='th.optim')
gin.external_configurable(optim.Adamax, module='th.optim')
gin.external_configurable(optim.ASGD, module='th.optim')
gin.external_configurable(optim.LBFGS, module='th.optim')
gin.external_configurable(optim.RMSprop, module='th.optim')
gin.external_configurable(optim.Rprop, module='th.optim')
gin.external_configurable(optim.SGD, module='th.optim')


# Constants
gin.constant('th.float', th.float)
gin.constant('th.float16', th.float16)
gin.constant('th.float32', th.float32)
gin.constant('th.float64', th.float64)
gin.constant('th.int', th.int)
gin.constant('th.int8', th.int8)
gin.constant('th.int16', th.int16)
gin.constant('th.int32', th.int32)
gin.constant('th.int64', th.int64)
gin.constant('th.uint8', th.uint8)
