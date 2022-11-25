import logging

from ClusterTrellis import Ginkgo_likelihood as likelihood

from .utils import get_logger
logger = get_logger(level=logging.WARNING)

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

# def gen_data(batch_size, max_length=10, test=False):
#     length = np.random.randint(1, max_length + 1)
#     x = np.random.randint(1, 100, (batch_size, length))
#     y = np.max(x, axis=1)
#     x, y = np.expand_dims(x, axis=2), np.expand_dims(y, axis=1)
#     return x, y


class EncoderSmallDeepSet(nn.Module):
    def __init__(self, pool="max", inFeatures=4, outFeatures=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=inFeatures, out_features=outFeatures),
            nn.ReLU(),
            nn.Linear(in_features=outFeatures, out_features=outFeatures),
            nn.ReLU(),
            nn.Linear(in_features=outFeatures, out_features=outFeatures),
            nn.ReLU(),
            nn.Linear(in_features=outFeatures, out_features=outFeatures),
        )
        # self.dec = nn.Sequential(
        #     nn.Linear(in_features=outFeatures, out_features=outFeatures),
        #     nn.ReLU(),
        #     nn.Linear(in_features=outFeatures, out_features=1),
        # )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            # for i in range(0,x.shape[1],4)
            x = x.sum(dim=1)

        # x = self.dec(x)
        return x




class DecoderSmallDeepSet(nn.Module):
    def __init__(self, outFeatures=8
):
        super().__init__()

        self.outfeatures= outFeatures
        self.dec = nn.Sequential(
            nn.Linear(in_features=outFeatures, out_features=outFeatures),
            nn.ReLU(),
            nn.Linear(in_features=outFeatures, out_features=1),
        )

    def forward(self, x):
        x = self.dec(x)
        return x




#-----------------------------------------

class amortized_fit(nn.Module):
    def __init__(self, hidden_size, output_size=1, batch_size=1, runLSTM=True, root_leaves = None, bidirectional=False, device=None):
        super(amortized_fit, self).__init__()
        self.runLSTM=runLSTM
        self.bidirectional= bidirectional
        self.root_leaves=root_leaves
        self.hidden_size = hidden_size
        self.batch_size =batch_size # batch_size=2, one for the node and one for the complement
        self.device=device
        # self.level=level

        # self.embedding = nn.Embedding(output_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        # self.gru = nn.GRU(hidden_size + 2, hidden_size, bidirectional=bidirectional) # (input, output ) dimensions. We we pass the [level, position in the 2^(N-1)-1 for loop] as input, by concatenating it to each of the input vectors with the previous output
        # self.LSTM = nn.LSTM(hidden_size + 2, hidden_size, bidirectional=bidirectional)

        if self.bidirectional:
            self.gru = nn.GRU(2*hidden_size + 2, hidden_size, bidirectional=bidirectional)  # (input, output ) dimensions. We we pass the [level, position in the 2^(N-1)-1 for loop] as input, by concatenating it to each of the input vectors with the previous output
            self.LSTM = nn.LSTM(2*hidden_size + 2, hidden_size, bidirectional=bidirectional)
            self.NiN1 = nn.Linear(2*hidden_size, 2*hidden_size)
            self.NiN2 = nn.Linear(2*hidden_size+ 2, 2*hidden_size+ 2)
            self.NiN_hidden1 = nn.Linear( hidden_size,  hidden_size)
            self.NiN_hidden2 = nn.Linear( hidden_size,  hidden_size)
            self.out = nn.Linear(2*hidden_size, output_size)
        else:
            self.gru = nn.GRU(hidden_size + 2, hidden_size,  bidirectional=bidirectional)  # (input, output ) dimensions. We we pass the [level, position in the 2^(N-1)-1 for loop] as input, by concatenating it to each of the input vectors with the previous output
            self.LSTM = nn.LSTM(hidden_size + 2, hidden_size, bidirectional=bidirectional)
            self.NiN1 = nn.Linear(hidden_size, hidden_size)
            self.NiN2 = nn.Linear(hidden_size, hidden_size)
            self.NiN_hidden1 = nn.Linear(hidden_size, hidden_size)
            self.NiN_hidden2 = nn.Linear(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)



        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.decoder_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # self.zi = nn.Linear(hidden_size, output_size)

    def initHidden(self):
        if self.bidirectional:
            return torch.ones(2,self.batch_size, self.hidden_size, device=self.device)
        else:
            return torch.ones(1,self.batch_size, self.hidden_size, device=self.device)

    def initInput(self):
        #     return torch.zeros(1,self.batch_size, self.hidden_size, device=self.device)
        if self.bidirectional:
            return torch.ones(1,self.batch_size, 2*self.hidden_size, device=self.device)
        else:
            return torch.ones(1,self.batch_size, self.hidden_size, device=self.device)


    def forward(self, a_trellis, input_node, treeLLH, treeProb, variational_tensors, exact_tensors, level=0):
        """ Individual tree forward pass"""
        location=0
        # for _ in range(Ntrees):
        #     treeLLH = []
        #     treeProb = []
        #     # treeProbFit = torch.ones(len(self.root.elements)-1, requires_grad=True)
        #     treeProbFit =[]
        #
        #     root= trellis.root

            # def fit_encDecTreeLH(self, input_node, a_trellis, treeLLH, treeProb, treeProbFit):

        if input_node.is_leaf():
            return

        Nleaves = len(input_node.elements)

        if input_node == a_trellis.root:
            decoder_input = self.initInput()
            # print("Device=", decoder_input.get_device())
            decoder_input = torch.cat((decoder_input, torch.tensor([level/(self.root_leaves-2), 0],device=self.device).view(1, self.batch_size, -1)),2)
        else:
            # Concat the [level, position in the 2^(N-1)-1 for loop] as input, by concatenating it to each of the input vectors with the previous output
            left_right = torch.cat((input_node.hidden, torch.tensor([level/(self.root_leaves-2),location/(2 ** (Nleaves - 1) - 1)],device=self.device)),0)
            # decoder_input = torch.stack([left_right,left_right], dim=0)
            # decoder_input = decoder_input.view(1, self.batch_size, -1)
            decoder_input = left_right.view(1, self.batch_size, -1)

        # if input_node == a_trellis.root:
        #     decoder_input = self.initInput()
        # else:
        #     # to implement a bidirectional NN (The hidden feature is initialized below)
        #     decoder_input = torch.stack([input_node.hidden, input_node.hidden], dim=0)
        #     decoder_input = decoder_input.view(1, self.batch_size, -1)

        # for _ in range(1):
        #
        #     # x = encoder(input)
        #
        #     # print("x=", x)
        #     # print("x shape = ", x.shape)
        #
        #     # decoder_hidden = x

        decoder_hidden = self.initHidden()
        cn=self.initHidden()
        # decoder_input = decoder.initInput()

        # print("input shape = ", decoder_input.size())
        # print("hidden shape = ",  decoder_hidden.size())

        for i in range(2 ** (Nleaves - 1) - 1):

            # output = self.embedding(input).view(1, 6, -1)
            # output = F.relu(output)
            """Bidirectional RNN - hidden=(num_layers * num_directions, batch, hidden_size)"""
            # hidden = hidden.view(1, -1, self.hidden_size)
            # input = input.view(1, self.batch_size, -1)

            # print("input shape = ", decoder_input.size())
            # print("hidden shape = ",  decoder_hidden.size())


            # print("Nleaves = ",Nleaves," | level", level," | pos = ",i," | Decoder input =  ", decoder_input)
            logger.debug("decoder_input shape = %s", decoder_input.size())
            logger.debug("decoder_hidden shape = %s", decoder_hidden.size())

            if self.runLSTM:
                decoder_output, (decoder_hidden, cn) = self.LSTM(decoder_input, (decoder_hidden, cn))
            else:
                decoder_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)

            # print("Output shape = ", decoder_output.size())
            # print("decoder_hidden shape = ", decoder_hidden.size())
            # print("cn shape = ", cn.size())

            """NiN for hidden vectors"""
            decoder_hidden = self.NiN_hidden1(decoder_hidden)
            decoder_hidden = self.relu(decoder_hidden)

            cn = self.NiN_hidden2(cn)
            cn = self.relu(cn)

            ### Use output as new input. If commented, then use same input to all sets of {node,complement}
            ### we pass the [level, position in the 2^(N-1)-1 for loop] as input, by concatenating it to each of the input vectors with the previous output
            xy_loc =  torch.tensor([level/(self.root_leaves-2),(i+1)/(2 ** (Nleaves - 1) - 1)],device=self.device).view(1, self.batch_size, -1)
            decoder_input = torch.cat((decoder_output, xy_loc),2)
            # print("decoder_input shape = ", decoder_input.size())

            # Add NiN layer for the input
            # decoder_input = self.NiN2(decoder_input)
            # decoder_input = self.relu(decoder_input)

            # z_i = self.softmax(self.out(output))
            # print("output = ", output)

            # we run {node, complement} as a batch of size=2. Then we sum them to get the partition function of the current vertex
            # zi_hidden = torch.sum(decoder_output, 1) #Sum vectors
            zi_hidden = decoder_output.view(1,-1) #concatenate vectors
            logger.info("zi_hidden dimension = ",zi_hidden.size())

            zi_hidden = self.NiN1(zi_hidden)
            zi_hidden = self.relu(zi_hidden)

            # zi_hidden = self.NiN2(zi_hidden)
            # zi_hidden = self.relu(zi_hidden)


            zi = self.out(zi_hidden)

            # print("decoder_input  = ", decoder_input)
            # print("decoder hidden", decoder_hidden)
            # zi, decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            # print("decoder_output  = ", decoder_output)
            if i == 0:
                output = zi_hidden
                Zvec = zi
            else:
                output = torch.cat((output, zi_hidden), 0)
                Zvec = torch.cat((Zvec, zi), 1)
            # print("i===",i)
            """ outputs saves the zi of the nodes and complements.
            hidden saves the hidden vectors of the nodes and complements"""



        logger.debug("-----------------")
        logger.debug("Zvec = %s", Zvec)
        logger.debug("output = %s", output)

        # print("Zvec=",Zvec)

        # we run {node, complement} as a batch of size=2
        """ Generated Z vector for input node"""
        logger.debug("LevelLHweightFit dim = %s", torch.flatten(Zvec).size())
        input_node.levelLHweightFit = self.softmax(torch.flatten(Zvec))


        """ Now sample"""
        special_element = list(input_node.elements)[0]
        logger.debug(f"special_element = {special_element}")

        """ Get nodes containing element within current subtree"""
        special_nodes = a_trellis.get_nodes_containing_element(input_node, special_element)

        """Root node can't be MAP (its the parent node => can't be a children node), so remove self from special_nodes"""
        special_nodes.remove(input_node)

        logger.debug(f"root  = {input_node}")
        logger.debug(f"special nodes = {special_nodes}")
        logger.debug(f"root.levelLHweight = {input_node.levelLHweight}")

        """Sample a node at current level following the likelihhood of each of them"""
        node_id = list(torch.utils.data.WeightedRandomSampler(input_node.levelLHweight, 1, replacement=True))[0]
        node = special_nodes[node_id]
        complement = a_trellis.get_complement_node(node, input_node.elements)
        logger.debug(f"node = {node}")
        logger.debug(f"complement = {complement}")

        treeProb.append(input_node.levelLHweight[node_id])
        exact_tensors.append(torch.tensor(input_node.levelLHweight).float())

        # Outputs has a batch size=2, 1 row for the node and 1 for the complement
        variational_tensors.append((input_node.levelLHweightFit, node_id))
        # print("Exact weights = ", torch.tensor(input_node.levelLHweight))
        # print("Learned weights = ", input_node.levelLHweightFit)

        """ Get difference between zi tensors"""
        # zi_diff.append(np.subtract(input_node.levelLHweight, input_node.levelLHweightFit))

        # Set node and complement hidden vectors (Each entry of the hidden vector). We repeat it so that we run the batch of the grandchildren {node,complement} together
        node.hidden = output[node_id]
        complement.hidden = output[node_id]
        # print("node.hidden = ", node.hidden)
        # print("===="*10)


        """ Get llh for the join of the pair {node,comlement} sampled"""
        # split_llh = likelihood.split_logLH(node.map_momentum,
        #                                    node.map_delta,
        #                                    complement.map_momentum,
        #                                    complement.map_delta,
        #                                    self.delta_min,
        #                                    self.lam)
        split_llh = input_node.get_energy_of_split(node, complement)
        treeLLH.append(split_llh)

        input_node.sample_features = input_node.compute_map_features(node, complement)



        """ Recursively repeat for the next level"""
        self.forward(a_trellis, node, treeLLH, treeProb, variational_tensors, exact_tensors, level=level+1)
        self.forward(a_trellis, complement, treeLLH, treeProb, variational_tensors,exact_tensors,  level=level+1)












######################################################################################
######################################################################################
####        OLD           ##################################################################################
######################################################################################
######################################################################################
######################################################################################

#---------------------------------
class DecoderRNN2(nn.Module):
    def __init__(self, hidden_size, output_size=1, batch_size=2, Nleaves=1,  device=None):
        super(DecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size =batch_size
        self.device=device

        # self.embedding = nn.Embedding(output_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.FC_layers={}
        for i in range(Nleaves-1):
            self.FC_layers[i]= nn.Linear(hidden_size, 2**(Nleaves-1)-1)

    def forward(self, input, hidden,level =0):
        # output = self.embedding(input).view(1, 6, -1)
        # output = F.relu(output)
        """hidden=(num_layers * num_directions, batch, hidden_size)"""
        hidden = hidden.view(1,-1, self.hidden_size)
        # print("input shape = ", input.shape)
        output, hidden = self.gru(input, hidden)
        # z_i = self.softmax(self.out(output))
        # z_i = self.out(output)
        z_i= self.FC_layers[level](output)
        # output = self.softmax(self.relu(self.out(output))+0.1)
        # output = output.view(-1,1)
        # print("output shape ",output.shape)
        return z_i, output, hidden

    def initHidden(self):
        return torch.ones(1, self.batch_size, self.hidden_size, device=self.device)
    def initInput(self):
    #     return torch.zeros(1,self.batch_size, self.hidden_size, device=self.device)
        return torch.ones(1,self.batch_size, self.hidden_size, device=self.device)




#-----------------------------------------

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size=1, batch_size=2, device=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size =batch_size
        self.device=device

        # self.embedding = nn.Embedding(output_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        # self.zi = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # output = self.embedding(input).view(1, 6, -1)
        # output = F.relu(output)
        """hidden=(num_layers * num_directions, batch, hidden_size)"""
        hidden = hidden.view(1,-1, self.hidden_size)
        input = input.view(1, self.batch_size,-1)
        # print("input shape = ", input.shape)
        # print("hidden shape = ", hidden.shape)
        # print("input = ", input)
        output, hidden = self.gru(input, hidden)
        # z_i = self.softmax(self.out(output))
        # print("output = ", output)

        # we run {node, complement} as a batch of size=2. Then we sum them to get the partition function of the current vertex
        zi_hidden = torch.sum(output, 1)
        # print("zi_hidden = ",zi_hidden)

        z_i = self.out(zi_hidden)

        # print("z_i=",z_i)

        # output = self.softmax(self.relu(self.out(output))+0.1)
        # output = output.view(-1,1)
        # print("output shape ",output.shape)
        return z_i, output, hidden

    def initHidden(self):
        return torch.ones(1, self.batch_size, self.hidden_size, device=self.device)
    def initInput(self):
    #     return torch.zeros(1,self.batch_size, self.hidden_size, device=self.device)
        return torch.ones(1,self.batch_size, self.hidden_size, device=self.device)



# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size=1, batch_size=2, device=None):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.batch_size =batch_size
#         self.device=device
#
#         # self.embedding = nn.Embedding(output_size, hidden_size)
#         # self.gru = nn.GRU(hidden_size, hidden_size)
#         self.gru = nn.GRU(1, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         # output = self.embedding(input).view(1, 6, -1)
#         # output = F.relu(output)
#         hidden = hidden.view(1,-1, self.hidden_size)
#         # print("hidden shape = ", hidden.shape)
#         output, hidden = self.gru(input, hidden)
#         output = self.softmax(self.out(output))
#         # output = output.view(-1,1)
#         # print("output shape ",output.shape)
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)
#     def initInput(self):
#     #     return torch.zeros(1,self.batch_size, self.hidden_size, device=self.device)
#         return torch.ones(1,self.batch_size, 1, device=self.device)







# class SmallSetTransformer(nn.Module):
#     def __init__(self,):
#         super().__init__()
#         self.enc = nn.Sequential(
#             SAB(dim_in=1, dim_out=64, num_heads=4),
#             SAB(dim_in=64, dim_out=64, num_heads=4),
#         )
#         self.dec = nn.Sequential(
#             PMA(dim=64, num_heads=4, num_seeds=1),
#             nn.Linear(in_features=64, out_features=1),
#         )

#     def forward(self, x):
#         x = self.enc(x)
#         x = self.dec(x)
#         return x.squeeze(-1)




# def train(encoder, decoder, learning_rate = 1e-4, deepsets=False, RNN=False):
# #     model = model.cuda()
# #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#     encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
#
#     criterion = nn.L1Loss()
#     losses = []
#     for _ in range(4):
#         x, y = load_data()
#         print("X shape = ", x.shape)
#         print("y shape = ", y.shape)
# #         x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda()
#         x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
#         x = encoder(x)
#         print("Encoded X shape = ", x.shape)
#         if deepsets:
#             loss = criterion(decoder(x), y)
#         if RNN:
#             decoder_hidden = decoder.initHidden()
#             print("decoder_hidden shape = ", decoder_hidden.shape)
#             decoder_output, decoder_hidden = decoder( x, decoder_hidden)
#             loss = criterion(decoder_output, y)
#
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()
#         loss.backward()
#         encoder_optimizer.step()
#         decoder_optimizer.step()
#         losses.append(loss.item())
#
#     return losses

