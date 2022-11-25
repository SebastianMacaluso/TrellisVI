import os
import pickle
import string
import time
import logging
import numpy as np

import logging

from .utils import get_logger
logger = get_logger(level=logging.WARNING)
# import importlib
from . import arch as net
import sys
from scipy.special import logsumexp, softmax
from torch.autograd import Variable



def load_data():
    indir = "data/invMassGinkgo/"
    NleavesMin=9
    in_filename = os.path.join(indir, "leaves_test_"+str(NleavesMin)+"_jets.pkl")
    with open(in_filename, "rb") as fd:
        x,y = pickle.load(fd, encoding='latin-1')
    return np.asarray(x), np.asarray(y)

#
#
import torch
import torch.nn as nn
# from modules import SAB, PMA
import numpy as np
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# inFeatures = 4
# outFeatures = 8
# batch_size = 2



def runDecRNN(encoder, decoder,decoder_input,  batch_size = 1, Nleaves=4):

    # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


    # encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    # print("# of encoder parameters =",encoder_params)
    # print("# of decoder parameters =",decoder_params)

    # losses = []
    # m = torch.nn.Softmax()
    # print("input", input)

    # input = torch.from_numpy(np.asarray(input)).float().view(batch_size, -1, 4)
    # print("----"*10)

    # print("input", input)
    # print("input", input.shape)
    # losses = []

    for _ in range(1):

        # x = encoder(input)

        # print("x=", x)
        # print("x shape = ", x.shape)

        # decoder_hidden = x

        decoder_hidden = decoder.initHidden()
        # decoder_input = decoder.initInput()

        for i in range(2**(Nleaves-1)-1):

            # print("decoder_input  = ", decoder_input)
            # print("decoder hidden", decoder_hidden)
            zi, decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # print("decoder_output  = ", decoder_output)
            if i==0:
                Zvec = zi
                output = decoder_output
            else:
                Zvec = torch.cat((Zvec,zi),1)
                output = torch.cat((output,decoder_output),0)
            # print("i===",i)
            """ outputs saves the zi of the nodes and complements
            hidden saves the hidden vectors of the nodes and complements"""
            ### Use output as new input. If commented, then use same input to all sets of {node,complement}
            decoder_input = decoder_output

        # print("-----------------")
        # print("Zvec = ", Zvec)
        # print("output = ", output)


    return Zvec, output

# def repackage_hidden(h):
#     """Wraps hidden states in new Variables, to detach them from their history."""
#     if type(h) == Variable:
#         return Variable(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)



def repackage_hidden(tensors):
    # """Wraps hidden states in new Variables, to detach them from their history."""
    # if type(h) == Variable:
    #     return Variable(h.data)
    # else:
    #     return tuple(repackage_hidden(v) for v in h if v.shape[0]>0)
    tensors = [[Variable(tensors[k][i].data, requires_grad=True) for i in range(len(tensors[k]))] for k in
                 range(len(tensors))]
    return tensors

def get_loss(encoder, decoder,sampled_tensors, nodes_id,learning_rate = 5e-1):

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    losses = []
    m = torch.nn.Softmax()

    # print("sampled_tensors shape =" , np.asarray(sampled_tensors).shape)
    # print("sampled_tensors =", sampled_tensors)
    # print("nodes_id = ", nodes_id)

    for _c in range(40):

        logLH = np.asarray([[m(sampled_tensors[k][i])[nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
                 range(len(sampled_tensors))])

        fitlogLH = np.prod(logLH, axis=1)

        # loss = criterion(input_tensors, target_tensor)
        loss = - 1/len(fitlogLH)*np.sum(np.log(fitlogLH))
        # loss = np.sum(fitlogLH)
        # print('Loss = ', loss)

        # sampled_tensors = repackage_hidden(sampled_tensors)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # print("Epoch # = ",_c)
        # loss.backward()
        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()
        losses.append(loss.item())



    return losses

#####----------------





#----------------------------

def trainEncDec2(encoder, decoder, input, nodes_id,level_size, batch_size = 1,learning_rate = 5e-2, Nleaves=4,LevelWeights = None, deepsets=False, RNN=False):

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("# of encoder parameters =",encoder_params)
    print("# of decoder parameters =",decoder_params)

    losses = []
    m = torch.nn.Softmax()
    # print("input", input)

    input = torch.from_numpy(np.asarray(input)).float().view(batch_size, -1, 4)
    print("----"*10)
    # print("input", input)
    print("input", input.shape)
    losses = []

    for _ in range(100):

        # x = encoder(input)

        # print("x=", x)
        # print("x shape = ", x.shape)

        # decoder_hidden = x
        decoder_hidden = decoder.initHidden()
        decoder_input = decoder.initInput()
        level =0
        for i in range(Nleaves-1):

            # print("decoder_input  = ", decoder_input)
            # print("decoder hidden", decoder_hidden)
            zi, decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, level=level)
            level+=1

            # print("decoder_output  = ", decoder_output)
            if i==0:
                outputs = zi
            else:
                outputs = torch.cat((outputs,zi),0)

            decoder_input = decoder_output


        # print("-----------------")
        # print("outputs b= ", outputs)
        # outputs = torch.abs(outputs).view(int(batch_size / (Nleaves - 1)), Nleaves - 1, -1)
        # outputs = outputs.view(batch_size , Nleaves - 1, -1)
        outputs = torch.transpose(outputs,0,1)
        # print("-----------------")
        # print("outputs a= ", outputs)


        logLH = np.asarray(
            [[m(outputs[k][i,0:int(level_size[k][i])])[nodes_id[k][i]] for i in range(len(outputs[k]))] for k in
             range(len(outputs))])



        fitlogLH = np.prod(logLH, axis=1)
        # print("fitlogLH = ", fitlogLH)

        # print("np.log(fitlogLH)=", np.log(fitlogLH))
        loss = - 1 / len(fitlogLH) * np.sum(np.log(fitlogLH))



        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        losses.append(loss.item())



    return losses






#--------------


def trainEncDec(encoder, decoder, input, nodes_id,level_size, batch_size = 1,learning_rate = 5e-2, Nleaves=4,LevelWeights = None, deepsets=False, RNN=False):

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("# of encoder parameters =",encoder_params)
    print("# of decoder parameters =",decoder_params)

    losses = []
    m = torch.nn.Softmax()
    # print("input", input)

    input = torch.from_numpy(np.asarray(input)).float().view(batch_size, -1, 4)
    print("----"*10)
    # print("input", input)
    print("input", input.shape)
    losses = []

    for _ in range(1):

        # x = encoder(input)

        # print("x=", x)
        # print("x shape = ", x.shape)

        # decoder_hidden = x
        decoder_hidden = decoder.initHidden()
        decoder_input = decoder.initInput()

        for i in range(2**(Nleaves-1)-1):

            # print("decoder_input  = ", decoder_input)
            # print("decoder hidden", decoder_hidden)
            zi, decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # print("decoder_output  = ", decoder_output)
            if i==0:
                outputs = zi
            else:
                outputs = torch.cat((outputs,zi),2)

            decoder_input = decoder_output


        # print("-----------------")
        print("outputs b= ", outputs)
        # outputs = torch.abs(outputs).view(int(batch_size / (Nleaves - 1)), Nleaves - 1, -1)
        outputs = outputs.view(int(batch_size / (Nleaves - 1)), Nleaves - 1, -1)
        print("-----------------")
        print("outputs a= ", outputs)

        logLH = np.asarray(
            [[m(outputs[k][i,0:int(level_size[k][i])])[nodes_id[k][i]] for i in range(len(outputs[k]))] for k in
             range(len(outputs))])

        # logLH = np.asarray(
        #     [[m(outputs[k])[i][nodes_id[k][i]] for i in range(len(outputs[k]))] for k in
        #      range(len(outputs))])
        # print("logLH = ", logLH)

        fitlogLH = np.prod(logLH, axis=1)
        # print("fitlogLH = ", fitlogLH)

        # print("np.log(fitlogLH)=", np.log(fitlogLH))
        loss = - 1 / len(fitlogLH) * np.sum(np.log(fitlogLH))



        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        losses.append(loss.item())



    return losses







def fit(input_tensor, target_tensor,learning_rate = 1e-3):
#     model = model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)

    criterion = nn.MSELoss()
    losses = []
    for _ in range(2000):


        loss = criterion(input_tensor, target_tensor)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses




def fitLogLH(input_tensors,sampled_tensors,nodes_id,learning_rate = 5e-2):
    """Fit the parametrized trellis using maximum likelihood loss
    Args:
        input_tensors: dictionary with all the pytorch tensors for the partial partition functions z_i of each vertex that we fit.
        sampled tensors: (tensor, idx) for each sampled tree from the variational distribution. The probability of a tree is obtained as
        prod_{i=root to tree leaves} tensor_i[idx]
        """
#     model = model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    optimizer = torch.optim.Adam(input_tensors, lr=learning_rate)
    # sampled_tensors2 = torch.tensor(sampled_tensors)
    # logger.info("Type of variational tensor = %s", sampled_tensors2[0][0].dtype)

    # criterion = nn.NLLLoss()
    losses = []

    m = torch.nn.Softmax()
    # In_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
    #      range(len(sampled_tensors))]
    # print("In_tensors == ", In_tensors)
    # print("----"*10)
    epochs=80
    for _c in range(epochs):

        logLH_list = np.asarray([[m(sampled_tensors[k][i])[nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
                 range(len(sampled_tensors))])
        #
        # logLH = [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
        #          range(len(sampled_tensors))]
        trees_logLH = np.prod(logLH_list, axis=1)
        # fitlogLH = logLH[:,0]*0.2*np.prod(logLH[:,1::], axis=1)

        #
        # logLH =[[input_tensors[k][i][nodes_id[k][i]] for i in range(len(input_tensors[k]))]for k in range(len(input_tensors)) ]
        # loss = criterion(input_tensors, target_tensor)
        loss = - 1/len(trees_logLH)*np.sum(np.log(trees_logLH))
        # loss = np.sum(fitlogLH)
        # print('Loss = ', loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    #
    # out_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
    #      range(len(sampled_tensors))]
    # print("out_tensors == ", out_tensors)
    # print("----"*10)

    return losses


# def fitLogLH_amortized(optimizer, my_lr_scheduler,  sampled_tensors, nodes_id, learning_rate=5e-2):
#     """Fit the parametrized trellis using maximum likelihood loss
#     Args:
#         input_tensors: dictionary with all the pytorch tensors for the partial partition functions z_i of each vertex that we fit.
#         sampled tensors: (tensor, idx) for each sampled tree from the variational distribution. The probability of a tree is obtained as
#         prod_{i=root to tree leaves} tensor_i[idx]
#         """
#     #     model = model.cuda()
#     #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#     # decoder_params = sum(p.numel() for p in model.parameters if p.requires_grad)
#     # logger.info("# of decoder parameters = %s", decoder_params)
#
#     # optimizer = torch.optim.Adam(model.parameters, lr=learning_rate)
#     logger.debug("Type of variational tensor = %s", sampled_tensors[0][0].dtype)
#     # torch.autograd.set_detect_anomaly(True)
#     # criterion = nn.NLLLoss()
#     losses = []
#
#     m = torch.nn.Softmax()  #"""We already applied the softmax, which maps logLH to LH"""
#     # In_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
#     #      range(len(sampled_tensors))]
#     # print("In_tensors == ", In_tensors)
#     # print("----"*10)
#     epochs = 1
#     for _c in range(epochs):
#         logger.debug(" sampled variational tensors before = %s", sampled_tensors)
#
#
#         LH_list = np.asarray(
#             [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
#              range(len(sampled_tensors))])
#
#         logger.debug("LH list = %s", LH_list)
#         # logLH_list = np.asarray(
#         #     [[m(sampled_tensors[k][i])[nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
#         #      range(len(sampled_tensors))])
#         #
#         # logLH = [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
#         #          range(len(sampled_tensors))]
#         trees_LH = np.prod(LH_list, axis=1)
#         logger.debug("Trees LH = %s", trees_LH)
#         # fitlogLH = logLH[:,0]*0.2*np.prod(logLH[:,1::], axis=1)
#
#         #
#         # logLH =[[input_tensors[k][i][nodes_id[k][i]] for i in range(len(input_tensors[k]))]for k in range(len(input_tensors)) ]
#         # loss = criterion(input_tensors, target_tensor)
#         loss = - 1 / len(trees_LH) * np.sum(np.log(trees_LH))
#         # loss = np.sum(fitlogLH)
#         logger.debug('Loss = %s', loss)
#
#         # print("Loss before =", loss)
#         # logger.info("Sampled tensors before = %s", sampled_tensors)
#         # sampled_tensors = repackage_hidden(sampled_tensors)
#         # logger.info("Sampled variational tensors after = %s", sampled_tensors)
#
#         """the gradient tensors are not reset unless we call model.zero_grad() or optimizer.zero_grad(). So if we take the gradient multiple times they get summed. We need to divide the loss by the number of steps in that case."""
#         optimizer.zero_grad()
#         # loss.backward(retain_graph=True)
#         """The computational graph is automatically destroyed when .backward() is called (unless retain_graph=True is specified)."""
#         loss.backward()
#         optimizer.step()
#         # losses.append(loss.item())
#         my_lr_scheduler.step()
#
#         # print("Loss after =", loss.item())
#         # print("---"*10)
#
#     #
#     # out_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
#     #      range(len(sampled_tensors))]
#     # print("out_tensors == ", out_tensors)
#     # print("----"*10)
#
#     # return losses
#     return loss.item()




def fitLogLH_amortized(optimizer, my_lr_scheduler,  sampled_tensors, nodes_id, learning_rate=5e-2, accumulation_steps=1):
    """Fit the parametrized trellis using maximum likelihood loss
    Args:
        input_tensors: dictionary with all the pytorch tensors for the partial partition functions z_i of each vertex that we fit.
        sampled tensors: (tensor, idx) for each sampled tree from the variational distribution. The probability of a tree is obtained as
        prod_{i=root to tree leaves} tensor_i[idx]
        """
    #     model = model.cuda()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # decoder_params = sum(p.numel() for p in model.parameters if p.requires_grad)
    # logger.info("# of decoder parameters = %s", decoder_params)

    # optimizer = torch.optim.Adam(model.parameters, lr=learning_rate)
    logger.debug("Type of variational tensor = %s", sampled_tensors[0][0].dtype)
    # torch.autograd.set_detect_anomaly(True)
    # criterion = nn.NLLLoss()
    losses = []

    m = torch.nn.Softmax()  #"""We already applied the softmax, which maps logLH to LH"""
    # In_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
    #      range(len(sampled_tensors))]
    # print("In_tensors == ", In_tensors)
    # print("----"*10)
    # epochs = 1
    # for _c in range(epochs):

    logger.debug(" sampled variational tensors before = %s", sampled_tensors)


    LH_list = np.asarray(
        [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
         range(len(sampled_tensors))])

    logger.debug("LH list = %s", LH_list)

    trees_LH = np.prod(LH_list, axis=1)
    logger.debug("Trees LH = %s", trees_LH)

    loss = - 1 / len(trees_LH) * np.sum(np.log(trees_LH))
    loss = loss/accumulation_steps

    logger.debug('Loss = %s', loss)

    loss.backward()
    # -------------
    if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()  # Now we can do an optimizer step
        model.zero_grad()  # Reset gradients tensors
    #-------------

    # print("Loss before =", loss)
    # logger.info("Sampled tensors before = %s", sampled_tensors)
    # logger.info("Sampled variational tensors after = %s", sampled_tensors)

    """the gradient tensors are not reset unless we call model.zero_grad() or optimizer.zero_grad(). So if we take the gradient multiple times they get summed. We need to divide the loss by the number of steps in that case."""
    # optimizer.zero_grad()
    # loss.backward(retain_graph=True)
    """The computational graph is automatically destroyed when .backward() is called (unless retain_graph=True is specified)."""
    # loss.backward()
    # optimizer.step()
    # losses.append(loss.item())
    my_lr_scheduler.step()

        # print("Loss after =", loss.item())
        # print("---"*10)

    return loss.item()







def eval_LogLH_amortized(sampled_tensors, nodes_id):
    """Fit the parametrized trellis using maximum likelihood loss
    Args:
        input_tensors: dictionary with all the pytorch tensors for the partial partition functions z_i of each vertex that we fit.
        sampled tensors: (tensor, idx) for each sampled tree from the variational distribution. The probability of a tree is obtained as
        prod_{i=root to tree leaves} tensor_i[idx]
        """
    #     model = model.cuda()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # decoder_params = sum(p.numel() for p in model.parameters if p.requires_grad)
    # logger.info("# of decoder parameters = %s", decoder_params)

    logger.debug("Type of variational tensor = %s", sampled_tensors[0][0].dtype)
    # torch.autograd.set_detect_anomaly(True)
    # criterion = nn.NLLLoss()
    losses = []

    m = torch.nn.Softmax()  # """We already applied the softmax, which maps logLH to LH"""
    # In_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
    #      range(len(sampled_tensors))]
    # print("In_tensors == ", In_tensors)
    # print("----"*10)
    epochs = 1
    for _c in range(epochs):
        logger.debug(" sampled variational tensors before = %s", sampled_tensors)

        LH_list = np.asarray(
            [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
             range(len(sampled_tensors))])

        logger.debug("LH list = %s", LH_list)
        # logLH_list = np.asarray(
        #     [[m(sampled_tensors[k][i])[nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
        #      range(len(sampled_tensors))])
        #
        # logLH = [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
        #          range(len(sampled_tensors))]
        trees_LH = np.prod(LH_list, axis=1)
        logger.debug("Trees LH = %s", trees_LH)
        # fitlogLH = logLH[:,0]*0.2*np.prod(logLH[:,1::], axis=1)

        #
        # logLH =[[input_tensors[k][i][nodes_id[k][i]] for i in range(len(input_tensors[k]))]for k in range(len(input_tensors)) ]
        # loss = criterion(input_tensors, target_tensor)
        loss = - 1 / len(trees_LH) * np.sum(np.log(trees_LH))
        # loss = np.sum(fitlogLH)
        logger.debug('Loss = %s', loss)

        # print("Loss before =", loss)
        # logger.info("Sampled tensors before = %s", sampled_tensors)
        # sampled_tensors = repackage_hidden(sampled_tensors)
        # logger.info("Sampled variational tensors after = %s", sampled_tensors)

        # optimizer.zero_grad()
        # # loss.backward(retain_graph=True)
        # loss.backward()
        # optimizer.step()
        # # losses.append(loss.item())
        # my_lr_scheduler.step()

        # print("Loss after =", loss.item())
        # print("---"*10)


    #
    # out_tensors = [[m(sampled_tensors[k][i]) for i in range(len(sampled_tensors[k]))] for k in
    #      range(len(sampled_tensors))]
    # print("out_tensors == ", out_tensors)
    # print("----"*10)

    # return losses
    return loss.item()
















##########################







if __name__=="__main__":


#
# import sys
# # !{sys.executable} -m pip install seaborn
# import seaborn as sns
# # %matplotlib inline
# import matplotlib.pyplot as plt
# sns.set(rc={"figure.figsize": (8, 4)}, style="whitegrid")


    RNNsize=outFeatures

    deepsets=False
    RNN = True


    X, Y = load_data()
    print("X shape = ", X.shape)
    print("y shape = ", Y.shape)
    #         x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()
    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    # Y=Y.view(-1,1)
    print("X.shape[1] = ", X.shape[1])

    # Set test target tensor Y
    # Y = torch.randint(low=0,high=2, size=(X.shape[0],X.shape[1]))


    if deepsets:
        models = [
    #     ("Set Transformer", SmallSetTransformer()),
    #     ("Deep Sets (max)", SmallDeepSet("max")),
    #     ("Deep Sets (mean)", SmallDeepSet("mean")),
        ("Deep Sets (sum)", net.EncoderSmallDeepSet("sum",inFeatures=inFeatures, outFeatures=outFeatures), net.DecoderSmallDeepSet(outFeatures=outFeatures)),
    ]
    if RNN:
        models = [
            ("Deep Sets (sum)", net.EncoderSmallDeepSet("sum", inFeatures=inFeatures, outFeatures=outFeatures),
             net.DecoderRNN(RNNsize,batch_size=batch_size, output_size = 1, device=device)),
        ]


    for _name, _encoder, _decoder in models:
        _losses = train(_encoder,_decoder, X, Y, deepsets=deepsets, RNN=RNN)
    #     plt.plot(_losses, label=_name)
    # plt.legend()
    # plt.xlabel("Steps")
    # plt.ylabel("Mean Absolute Error")
    # plt.yscale("log")
    # plt.show()

    print(_losses)