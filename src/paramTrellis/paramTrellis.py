
import itertools
import string
import numpy as np
import logging
import os

import wandb
import time
# import torch
# from torchsummary import summary

from . import utils
from .utils import get_logger
logger = get_logger(level=logging.WARNING)


from ClusterTrellis.HierarchicalTrellis import HierarchicalTrellis


from collections import namedtuple
import torch
# import encDec

# from hierarchical_trellis import HierarchicalTrellis
# from jet_node_invM import JetNode


#---------------------------
import torch
import torch.nn as nn
# from modules import SAB, PMA


from . import arch as net
from . import encDec as fit
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
inFeatures = 4
# outFeatures = 8
# batch_size = 1
# RNNsize=outFeatures

# embed = False
deepsets=False
RNN = True



class paramTrellis(HierarchicalTrellis):
    """Class to define the node pairwise splitting energy and calculate node features for a given model """

    def __init__(self, *args, **kargs):
        super().__init__()
        self.VI_params = []
        self.VI_exact_params =[]
        # self.model_dir = "experiments/model_checkpoints/"


    #
    # def compute_map_tree(self):
    #     """Finf the MLE (MAP) tree, and compute its momentums, deltas, and energies."""
    #
    #     """Start from the leaf nodes"""
    #     for leaf in self.leaves():
    #         """ Compute map tree where each leaf is the root"""
    #         logger.debug(f"Leaf == {leaf}")
    #         leaf.compute_map_tree_split(self)
    #
    #     """Set map tree for all inner nodes in order from smallest to largest number of elements.
    #     Start from the leaves"""
    #     elements = list(self.elements)
    #     logger.debug(f"elements = {elements}")
    #     for i in range(2, len(elements) + 1):
    #         for elem in itertools.combinations(elements, i):
    #             logger.debug(f"element = {elem}")
    #             elem = frozenset(elem)
    #             """Find node for each trellis label"""
    #             node = self.elements_2_node[elem]
    #
    #             """ Compute map tree where the current node is the root"""
    #             node.compute_map_tree_split(self)
    #             # node.initialize_paramNodeTensors(self)
    #
    #     logger.info(f"====="*10)
    #     logger.debug(f"Assigned MAP values for {self, self.root.map_tree_energy, self.root.map_children, self.root.map_features}")
    #     logger.info(f"---------"*10)
    #     logger.info(f"Partition function (logLH) = {self.root.logZ}")
    #     logger.info(f"MLE (LH, LLH) = {np.exp(self.map_tree_energy), self.map_tree_energy}")
    #     logger.info(f"Tot trees # =  {self.root.N_trees}")
    #
    #
    #     return self.map_tree_energy, self.root.logZ, self.root.N_trees



    def MLHfit(self,root, Ntrees):
        """ Maximum likelihood hierarchy fit: Fit the variational trellis parameters sampling from the exact trellis and using maximum likelihood.
        Initialize every z_i as a torch tensor"""
        treedist =[]
        treesProb = []
        # treesProbFit = []
        trees_variational_tensors=[]
        nodes_id=[]
        logger.info("VI Exact params = %s", np.asarray(self.VI_exact_params))
        logger.info("----" * 10)
        m = torch.nn.Softmax()
        logger.info("VI_params = %s", np.asarray([m(entry["params"][0].detach()).numpy() for entry in self.VI_params]))

        for _ in range(Ntrees):
            treeLLH = []
            treeProb = []
            # treeProbFit = torch.ones(len(self.root.elements)-1, requires_grad=True)
            # treeProbFit =[]
            variational_tensors=[]

            root.sample_distributions(self,root,treeLLH, treeProb,variational_tensors,0)
            # print("treeProbFit === ", treeProbFit)

            # print("Trellis prob = ", treeProb)

            treedist.append(np.sum(treeLLH))
            treesProb.append(np.prod(treeProb))
            trees_variational_tensors.append([x for (x,y) in variational_tensors])
            nodes_id.append([y for (x,y) in variational_tensors])
            # treesProbFit.append(treeProbFit)

            # print("treeProbFit 2=== ", treeProbFit)

            logger.debug(f"treeLLH = {treeLLH}")

            # _losses = encDec.fit(treesProbFit, treesProb)

        # logger.info(f"treedist = {treedist}")
        #
        # # treesProbFit = torch.tensor(treesProbFit, requires_grad=True).view(-1,1)
        # treesProbFit = torch.prod(treeProbFit).view(-1,1)
        # print("treeProbFit 2=== ", treesProbFit)
        #
        _losses = fit.fitLogLH(self.VI_params,trees_variational_tensors,nodes_id)
        # # _losses=0


        logger.info("VI_params after = %s", np.asarray([m(entry["params"][0].detach()).numpy() for entry in self.VI_params]))
        logger.info("----"*10)
        return np.asarray(treedist),treesProb, trees_variational_tensors, _losses





    def train_amortized_MLH(self,Ntrees, epochs, lr, accumulation_steps, model_dir, RNNsize=8 ,batch_size=1,  bidirectional=False, restore_file=None):

        StartTime = time.time()

        def get_lr(optimizer):
            lr=[]
            for param_group in optimizer.param_groups:
                lr+=[param_group['lr']]
            return lr

        model = net.amortized_fit(RNNsize, output_size=1, batch_size=batch_size, root_leaves = len(self.root.elements), bidirectional=bidirectional, device=device).cuda() if torch.cuda.is_available() else net.amortized_fit(RNNsize, output_size=1, batch_size=batch_size, root_leaves = len(self.root.elements), bidirectional=bidirectional, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.96)
        # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.88)
        # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

        decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Training with GPU =", torch.cuda.is_available())
        print("# of decoder parameters =", decoder_params)
        print("NN Hidden size = ", RNNsize)
        print("Bidirectional = ", bidirectional)
        # print("Architecture = ", summary(model, (INPUT_SHAPE)))
        print("Architecture = ", model)
        logger.debug("Model parameters = %s", model.parameters)

        # reload weights from restore_file if specified
        print("restore_file = ", restore_file)
        if restore_file!="None":
            restore_path = os.path.join(model_dir, 'last.pth.tar')
            logging.info("Restoring parameters from {}".format(restore_path))
            print("Restoring parameters from =",restore_path)
            utils.load_checkpoint(restore_path, model, optimizer)


        model.train()

        # treedist = []
        # treesProb = []

        _losses=[]
        # TotaltreesProb =[]
        # epochs=300
        min_loss = np.inf
        for epoch in range(epochs):

            if epoch%20==0:
                print("Epoch = ", epoch, " | lr=",get_lr(optimizer))

            treedist = []
            treesProb = []
            # treesProbFit = []


            _loss=0
            for step in range(accumulation_steps):
                trees_variational_tensors = []
                # trees_zi_diff = []
                trees_exact_tensors = []
                nodes_id = []

                for _ in range(Ntrees):
                    treeLLH = []
                    treeProb = []
                    variational_tensors=[]
                    # zi_diff = []
                    exact_tensors = []

                    model(self, self.root, treeLLH, treeProb, variational_tensors, exact_tensors)

                    # treedist.append(np.sum(treeLLH))
                    treesProb.append(np.prod(treeProb))
                    trees_variational_tensors.append([x for (x,y) in variational_tensors])
                    # trees_zi_diff.append(zi_diff)
                    trees_exact_tensors.append(exact_tensors)
                    nodes_id.append([y for (x,y) in variational_tensors])

                    logger.debug(f"treeLLH = {treeLLH}")


                logger.debug(" sampled variational tensors before = %s", trees_variational_tensors)

                # loss_function = "MSE"
                loss_function = "MLE"
                if loss_function=="MLE":
                    LH_list = np.asarray(
                        [[trees_variational_tensors[k][i][nodes_id[k][i]] for i in range(len(trees_variational_tensors[k]))] for k in
                         range(len(trees_variational_tensors))])

                    logger.debug("LH list = %s", LH_list)

                    trees_LH = np.prod(LH_list, axis=1)
                    logger.debug("Trees LH = %s", trees_LH)

                    loss = - 1 / len(trees_LH) * np.sum(np.log(trees_LH))

                elif loss_function=="MSE":
                    # MSEloss = nn.MSELoss(reduction='mean')
                    MSEloss = nn.MSELoss()
                    # print(" Variational tensors shape = ", np.asarray(trees_variational_tensors).shape)
                    # print(" exact_tensors tensors shape = ", np.asarray(trees_exact_tensors).shape)
                    loss = 1 / len(trees_variational_tensors) * np.sum([1 / len(trees_variational_tensors[k]) * np.sum([MSEloss(trees_variational_tensors[k][i],trees_exact_tensors[k][i])  for i in range(len(trees_variational_tensors[k]))]) for k in range(len(trees_variational_tensors))])


                    LH_list = np.asarray(
                        [[trees_variational_tensors[k][i][nodes_id[k][i]] for i in range(len(trees_variational_tensors[k]))] for k in
                         range(len(trees_variational_tensors))])

                    logger.debug("LH list = %s", LH_list)

                    trees_LH = np.prod(LH_list, axis=1)
                    logger.debug("Trees LH = %s", trees_LH)

                    fitt_prob =  1 / len(trees_LH) * np.sum(np.log(trees_LH))



                    # print("Partial loss = ", MSEloss(trees_variational_tensors[0][0],trees_exact_tensors[0][0]))
                    # print("Semin partial loss = ", np.sum([MSEloss(trees_variational_tensors[0][i],trees_exact_tensors[0][i])  for i in range(len(trees_variational_tensors[0]))]))
                # 
                # print("Loss = ", loss)

                """the gradient tensors are not reset unless we call model.zero_grad() or optimizer.zero_grad(). So if we take the gradient multiple times they get summed. We need to divide the loss by the number of steps in that case."""
                loss = loss / accumulation_steps
                _loss+=loss.item()
                logger.debug('Loss = %s', loss)
                # print("Loss = ", loss)
                # print("Loss type= ", loss.dtype)
                """The computational graph is automatically destroyed when .backward() is called (unless retain_graph=True is specified)."""
                loss.backward()

            # print("Epoch = ", epoch)
            # print("-----"*10)
            #Update best loss
            is_best= _loss <=min_loss
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'loss':_loss},
                                  is_best=is_best,
                                  checkpoint=model_dir)
            """We accumulated the gradients of accumulation_steps times before updating. Thus, we use less memory to retain the computational graph"""
            # if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            model.zero_grad()  # Reset gradients tensors

            my_lr_scheduler.step()

            _losses.append(_loss)
            #-----------------------

            if loss_function == "MLE":
                print("Maximum Likelihood loss")
                wandb.log({'epoch': epoch, 'MLE_loss': _loss, 'Exact_prob': -1/len(treesProb)*sum(np.log(treesProb)),'Loss_prob_ratio':_loss/(-1/len(treesProb)*sum(np.log(treesProb))),'total_time': time.time() - StartTime, "learning_rate":get_lr(optimizer)})

            else:
                print("MSE loss")
                wandb.log({'MSE_epoch': epoch, 'MSE_loss': _loss, 'MSE_Exact_prob': 1/len(treesProb)*sum(np.log(treesProb)), 'Fitting_prob': fitt_prob,'Fitt_exact_prob_ratio':fitt_prob/(1/len(treesProb)*sum(np.log(treesProb))),'MSE_total_time': time.time() - StartTime, "learning_rate":get_lr(optimizer)})

        # logger.info("VI_params after = %s", np.asarray([m(entry["params"][0].detach()).numpy() for entry in self.VI_params]))
        logger.info("----"*10)
        if torch.cuda.is_available(): print("Training with GPU =", torch.cuda.is_available())
        print("DONE WITH TRAINING!")
        return treesProb, trees_variational_tensors, _losses, model
























    # def train_amortized_MLH(self,Ntrees, epochs, lr, accumulation_steps, model_dir, restore_file=None):
    #
    #     StartTime = time.time()
    #
    #     def get_lr(optimizer):
    #         lr=[]
    #         for param_group in optimizer.param_groups:
    #             lr+=[param_group['lr']]
    #         return lr
    #
    #     model = net.amortized_fit(RNNsize, output_size=1, batch_size=2, root_leaves = len(self.root.elements), device=device).cuda() if torch.cuda.is_available() else net.amortized_fit(RNNsize, output_size=1, batch_size=2, root_leaves = len(self.root.elements), device=device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    #     my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.98)
    #     # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.88)
    #     # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    #     # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    #
    #     decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print("# of decoder parameters =", decoder_params)
    #     logger.debug("Model parameters = %s", model.parameters)
    #
    #     # reload weights from restore_file if specified
    #     print("restore_file = ", restore_file)
    #     if restore_file!="None":
    #         restore_path = os.path.join(model_dir, 'last.pth.tar')
    #         logging.info("Restoring parameters from {}".format(restore_path))
    #         print("Restoring parameters from =",restore_path)
    #         utils.load_checkpoint(restore_path, model, optimizer)
    #
    #
    #     model.train()
    #
    #     # treedist = []
    #     # treesProb = []
    #
    #     _losses=[]
    #     # TotaltreesProb =[]
    #     # epochs=300
    #     min_loss = np.inf
    #     for epoch in range(epochs):
    #
    #         if epoch%20==0:
    #             print("Epoch = ", epoch, " | lr=",get_lr(optimizer))
    #
    #         treedist = []
    #         treesProb = []
    #         # treesProbFit = []
    #
    #
    #         _loss=0
    #         for step in range(accumulation_steps):
    #             trees_variational_tensors = []
    #             nodes_id = []
    #
    #             for _ in range(Ntrees):
    #                 treeLLH = []
    #                 treeProb = []
    #                 variational_tensors=[]
    #
    #                 model(self, self.root, treeLLH, treeProb, variational_tensors)
    #
    #                 # treedist.append(np.sum(treeLLH))
    #                 treesProb.append(np.prod(treeProb))
    #                 trees_variational_tensors.append([x for (x,y) in variational_tensors])
    #                 nodes_id.append([y for (x,y) in variational_tensors])
    #
    #                 logger.debug(f"treeLLH = {treeLLH}")
    #
    #
    #             logger.debug(" sampled variational tensors before = %s", trees_variational_tensors)
    #             LH_list = np.asarray(
    #                 [[trees_variational_tensors[k][i][nodes_id[k][i]] for i in range(len(trees_variational_tensors[k]))] for k in
    #                  range(len(trees_variational_tensors))])
    #
    #             logger.debug("LH list = %s", LH_list)
    #
    #             trees_LH = np.prod(LH_list, axis=1)
    #             logger.debug("Trees LH = %s", trees_LH)
    #
    #             loss = - 1 / len(trees_LH) * np.sum(np.log(trees_LH))
    #
    #             """the gradient tensors are not reset unless we call model.zero_grad() or optimizer.zero_grad(). So if we take the gradient multiple times they get summed. We need to divide the loss by the number of steps in that case."""
    #             loss = loss / accumulation_steps
    #             _loss+=loss.item()
    #             logger.debug('Loss = %s', loss)
    #             """The computational graph is automatically destroyed when .backward() is called (unless retain_graph=True is specified)."""
    #             loss.backward()
    #
    #         #Update best loss
    #         is_best= _loss <=min_loss
    #         # Save weights
    #         utils.save_checkpoint({'epoch': epoch + 1,
    #                                'state_dict': model.state_dict(),
    #                                'optim_dict': optimizer.state_dict(),
    #                                'loss':_loss},
    #                               is_best=is_best,
    #                               checkpoint=model_dir)
    #         """We accumulated the gradients of accumulation_steps times before updating. Thus, we use less memory to retain the computational graph"""
    #         # if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
    #         optimizer.step()  # Now we can do an optimizer step
    #         model.zero_grad()  # Reset gradients tensors
    #
    #         my_lr_scheduler.step()
    #
    #         _losses.append(_loss)
    #         #-----------------------
    #
    #
    #         wandb.log({'epoch': epoch, 'loss': _loss, 'Exact_prob': -1/len(treesProb)*sum(np.log(treesProb)),'Loss_prob_ratio':_loss/(-1/len(treesProb)*sum(np.log(treesProb))),'total_time': time.time() - StartTime, "learning_rate":get_lr(optimizer)})
    #
    #     # logger.info("VI_params after = %s", np.asarray([m(entry["params"][0].detach()).numpy() for entry in self.VI_params]))
    #     logger.info("----"*10)
    #     if torch.cuda.is_available(): print("Training with GPU =", torch.cuda.is_available())
    #     print("DONE WITH TRAINING!")
    #     return treesProb, trees_variational_tensors, _losses, model




    def eval_amortized_MLH(self,Ntrees, epochs, model, model_dir, restore_file=None):

        StartTime = time.time()

        # def get_lr(optimizer):
        #     lr=[]
        #     for param_group in optimizer.param_groups:
        #         lr+=[param_group['lr']]
        #     return lr
        # 
        # model = net.amortized_fit(RNNsize, output_size=1, batch_size=2, root_leaves = len(self.root.elements), device=device).cuda() if torch.cuda.is_available() else net.amortized_fit(RNNsize, output_size=1, batch_size=2, root_leaves = len(self.root.elements), device=device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
        # 
        # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

        decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("# of decoder parameters =", decoder_params)

        # reload weights from restore_file if specified
        if restore_file is not None:
            restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
            logging.info("Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

        model.eval()

        # treedist = []
        # treesProb = []

        _losses=[]
        # TotaltreesProb =[]
        # epochs=300

        for epoch in range(epochs):

            if epoch%50==0:
                print("Epoch = ",epoch)

            treedist = []
            treesProb = []
            # treesProbFit = []
            trees_variational_tensors = []
            trees_exact_tensors = []
            nodes_id = []

            for _ in range(Ntrees):
                treeLLH = []
                treeProb = []
                # treeProbFit = torch.ones(len(self.root.elements)-1, requires_grad=True)
                treeProbFit =[]
                variational_tensors=[]
                exact_tensors = []


                model(self, self.root, treeLLH, treeProb, variational_tensors,exact_tensors )



                treedist.append(np.sum(treeLLH))
                treesProb.append(np.prod(treeProb))
                trees_variational_tensors.append([x for (x,y) in variational_tensors])
                nodes_id.append([y for (x,y) in variational_tensors])
                # treesProbFit.append(treeProbFit)

                # print("treeProbFit 2=== ", treeProbFit)

                logger.debug(f"treeLLH = {treeLLH}")


            decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("# of decoder parameters = %s", decoder_params)

            logger.info("Model parameters = %s",model.parameters)
            # optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

            _loss = fit.eval_LogLH_amortized( trees_variational_tensors,nodes_id)
        # # _losses=0

            _losses.append(_loss)
            # TotaltreesProb+=treesProb

            # wandb.log({'Eval_epoch': epoch, 'Eval_loss': _loss, 'Eval_exact_prob': -1/len(treesProb)*sum(np.log(treesProb)),'Eval_loss_prob_ratio':_loss/(-1/len(treesProb)*sum(np.log(treesProb))),'Eval_total_time': time.time() - StartTime})
        print("EVALUATION RESULTS")
        print("Eval_exact_prob = ", -1/len(treesProb)*sum(np.log(treesProb)))
        print("Eval_loss = ",_losses)
        print("Eval_loss_prob_ratio = ",_loss/(-1/len(treesProb)*sum(np.log(treesProb))))
        print("Eval_total_time = ", time.time() - StartTime)
        # logger.info("VI_params after = %s", np.asarray([m(entry["params"][0].detach()).numpy() for entry in self.VI_params]))
        logger.info("----"*10)
        return np.asarray(treedist),treesProb, trees_variational_tensors, _losses




    #
    #
    #
    # def train_amortized_MLH(self,Ntrees, epochs, lr):
    #
    #     StartTime = time.time()
    #
    #     def get_lr(optimizer):
    #         lr=[]
    #         for param_group in optimizer.param_groups:
    #             lr+=[param_group['lr']]
    #         return lr
    #
    #     model = net.amortized_fit(RNNsize, output_size=1, batch_size=2, root_leaves = len(self.root.elements), device=device).cuda() if torch.cuda.is_available() else net.amortized_fit(RNNsize, output_size=1, batch_size=2, root_leaves = len(self.root.elements), device=device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    #     # my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)
    #     my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    #
    #     decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print("# of decoder parameters =", decoder_params)
    #
    #     model.train()
    #
    #     # treedist = []
    #     # treesProb = []
    #
    #     _losses=[]
    #     # TotaltreesProb =[]
    #     # epochs=300
    #
    #     for step in range(epochs*accumulation_steps):
    #
    #         if step%(accumulation_step*20)==0:
    #             print("Epoch = ",step/accumulation_step, " | lr=",get_lr(optimizer))
    #
    #         treedist = []
    #         treesProb = []
    #         # treesProbFit = []
    #         trees_variational_tensors = []
    #         nodes_id = []
    #
    #         for _ in range(Ntrees):
    #             treeLLH = []
    #             treeProb = []
    #             # treeProbFit = torch.ones(len(self.root.elements)-1, requires_grad=True)
    #             treeProbFit =[]
    #             variational_tensors=[]
    #
    #
    #             model(self, self.root, treeLLH, treeProb, variational_tensors)
    #
    #
    #
    #             treedist.append(np.sum(treeLLH))
    #             treesProb.append(np.prod(treeProb))
    #             trees_variational_tensors.append([x for (x,y) in variational_tensors])
    #             nodes_id.append([y for (x,y) in variational_tensors])
    #             # treesProbFit.append(treeProbFit)
    #
    #             # print("treeProbFit 2=== ", treeProbFit)
    #
    #             logger.debug(f"treeLLH = {treeLLH}")
    #
    #
    #         decoder_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #         logger.info("# of decoder parameters = %s", decoder_params)
    #
    #         logger.info("Model parameters = %s",model.parameters)
    #         # optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    #
    #         #--------------------
    #         # m = torch.nn.Softmax()
    #         logger.debug(" sampled variational tensors before = %s", sampled_tensors)
    #         LH_list = np.asarray(
    #             [[sampled_tensors[k][i][nodes_id[k][i]] for i in range(len(sampled_tensors[k]))] for k in
    #              range(len(sampled_tensors))])
    #
    #         logger.debug("LH list = %s", LH_list)
    #
    #         trees_LH = np.prod(LH_list, axis=1)
    #         logger.debug("Trees LH = %s", trees_LH)
    #
    #         loss = - 1 / len(trees_LH) * np.sum(np.log(trees_LH))
    #
    #         """the gradient tensors are not reset unless we call model.zero_grad() or optimizer.zero_grad(). So if we take the gradient multiple times they get summed. We need to divide the loss by the number of steps in that case."""
    #         loss = loss / accumulation_steps
    #         logger.debug('Loss = %s', loss)
    #         """The computational graph is automatically destroyed when .backward() is called (unless retain_graph=True is specified)."""
    #         loss.backward()
    #
    #         """We accumulate the gradients of accumulation_steps times before updating. Thus, we use less memory to retain the computational graph"""
    #         if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
    #             optimizer.step()  # Now we can do an optimizer step
    #             model.zero_grad()  # Reset gradients tensors
    #
    #             my_lr_scheduler.step()
    #
    #             _losses.append(loss.item())
    #             #-----------------------
    #
    #
    #
    #
    #     #
    #     #     _loss = fit.fitLogLH_amortized(optimizer,my_lr_scheduler, trees_variational_tensors,nodes_id)
    #     # # # _losses=0
    #     #
    #     #     _losses.append(_loss)
    #         # TotaltreesProb+=treesProb
    #
    #         wandb.log({'epoch': step/accumulation_step, 'loss': loss.item(), 'Exact_prob': -1/len(treesProb)*sum(np.log(treesProb)),'Loss_prob_ratio':_loss/(-1/len(treesProb)*sum(np.log(treesProb))),'total_time': time.time() - StartTime, "learning_rate":get_lr(optimizer)})
    #
    #     # logger.info("VI_params after = %s", np.asarray([m(entry["params"][0].detach()).numpy() for entry in self.VI_params]))
    #     logger.info("----"*10)
    #     if torch.cuda.is_available(): print("Training with GPU =", torch.cuda.is_available())
    #     print("DONE WITH TRAINING!")
    #     return np.asarray(treedist),treesProb, trees_variational_tensors, _losses, model