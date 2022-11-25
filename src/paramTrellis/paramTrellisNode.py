import numpy as np
import logging
from scipy.special import logsumexp, softmax

from ClusterTrellis.trellis_node import TrellisNode
"""Replace with model auxiliary scripts to calculate the energy function"""
from ClusterTrellis import Ginkgo_likelihood as likelihood

from .utils import get_logger
logger = get_logger(level=logging.WARNING)


# #---------------------------
import torch
import torch.nn as nn
# from modules import SAB, PMA
import numpy as np
#
from . import arch as net
from . import encDec
# #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #
inFeatures = 4





class paramTrellisNode(TrellisNode):
    def __init__(self,model_params,
                 elements = None,
                 children = None,
                 map_features = None):
        super().__init__(model_params,
                 elements,
                 children,
                 map_features)

        self.levelLHweightFit = []
        self.visited = False
        self.loss = None
        self.hidden = None

        self.N_powersets = 1
        self.sample_features = None
        # self.sample_momentum = None
        # self.sample_delta = None

    def compute_map_tree_split(self, a_trellis):
        """Compute the momentum, delta, and energies over the trellis.
        Save the MLE(MAP) energy split, marginal (partition function Z) and count number of trees.
        Assumes the map_energy of all the descendent nodes has already been computed!!"""
        logger.debug(f"Computing MAP tree rooted at {self}")


        """If the node is a leaf, set it to some default likelihood =1"""
        if self.is_leaf():
            self.map_tree_energy = 0
            self.logZ = 0.

            logger.debug(f"Leaf Z = {self.Z}")

        elif self.map_tree_energy is None and self.logZ is None:
            """Choose the 1st element of the parent node as a special element. 
            Get all sub-clusters that contain this element and their complement. 
            As this element has to be in one of the 2 sub-clusters that are the children, 
            then this way we cover all possible clusterings.
            For each node, it's "energy" is the llh of the join between the node and its complement.
            Find the max energy of all possible pairs of children for this parent node."""
            special_element = list(self.elements)[0]
            logger.debug(f"special_element = {special_element}")

            """ Get nodes containing element within current subtree"""
            special_nodes = a_trellis.get_nodes_containing_element(self, special_element)
            logger.debug(f"special nodes = {special_nodes}")

            """Root node can't be MAP (its the parent node => can't be a children node), so remove self from special_nodes"""
            special_nodes.remove(self)

            """Initialize paramNode Tensors"""
            self.levelLHweightFit = torch.rand(len(special_nodes), requires_grad=True, device=device)
            a_trellis.VI_params.append({"params": self.levelLHweightFit})


            self.map_tree_energy = -np.inf
            self.logZ = -np.inf

            N_trees = 0
            # N_powersets = 0
            N_powersets = len(special_nodes)

            for node in special_nodes:

                complement = a_trellis.get_complement_node(node, self.elements)
                logger.debug(f" complement = {complement}")

                if not node.visited:
                    N_powersets += node.N_powersets
                    node.visited = True
                if not complement.visited:
                    N_powersets += complement.N_powersets
                    complement.visited = True

                """Compute the pairing of nodes energy (llh)"""
                split_llh = self.get_energy_of_split(node, complement)

                """Add last splitting energy to subtree energy."""
                energy = split_llh + node.map_tree_energy + complement.map_tree_energy

                """Count number of trees, only if tree is allowed under the model"""
                if energy > -np.inf:
                    N_trees += node.N_trees * complement.N_trees

                """Compute partition function in a stable way
                logZ_i = log[LH(a, b)] + logZ_a + logZ_b
                logZ_p = scipy.misc.logsumexp(np.asarray([logZ_p, logZ_i]))."""

                partial_logZ = split_llh + node.logZ + complement.logZ
                self.logZ = logsumexp(np.asarray([self.logZ, partial_logZ]))

                self.levelLHweight.append(partial_logZ)

                logger.debug(f" Pair of nodes Z = { node.Z, complement.Z}")
                logger.debug(f" Pair of nodes Energy = { node.map_tree_energy, complement.map_tree_energy}")


                """Save if new value for energy is higher"""
                if energy > self.map_tree_energy:

                    self.map_features = self.compute_map_features(node, complement)
                    self.map_tree_energy = energy

                    """Save the 2 clusters of leaves that give the MAP (MLE). This is for each inner node. 
                    The we could start from the root of the tree and traverse down the tree."""
                    self.map_children = (node, complement)

            """Assign momentum and delta for nodes that are not possible under the model, i.e. give a deltaP<delta_min. 
            These nodes have a llh=-np.inf so they don't contribute to the MLE or Z, but we need them to have a complete trellis"""
            if self.map_tree_energy == - np.inf:
                complement = a_trellis.get_complement_node(special_nodes[0], self.elements)
                self.map_features = self.compute_map_features(special_nodes[0], complement)


            self.N_trees = N_trees
            self.N_powersets = N_powersets

            """Normalize"""
            logger.debug(f"level LH soft before = {self.levelLHweight}")

            if self.logZ == -np.inf:
                """Assign 0 probability when Delta_parent is below the treshold not allowed under the model"""
                self.levelLHweight = np.zeros(len(self.levelLHweight))

            else:
                """ Normalize to 1"""
                self.levelLHweight = softmax(np.asarray(self.levelLHweight))
                a_trellis.VI_exact_params.append(self.levelLHweight)

            logger.debug(f"---------" * 10)
            logger.debug(f"Special nodes= {special_nodes}")
            logger.debug(f"level LH soft after = {self.levelLHweight}")
            logger.debug("---------" * 10)
            logger.debug("---------" * 10)




    # def initialize_paramNodeTensors(self,a_trellis):
    #
    #     # node = special_nodes[0]
    #     # complement = a_trellis.get_complement_node(node, self.elements)
    #     # self.map_momentum = self.compute_momentum(node, complement)
    #     # self.map_delta = self.compute_delta(node, complement)
    #     #
    #     # if embed:
    #     #     pass
    #     #     # """This is the case if we use an encoder NN"""
    #     #     # leaves_p = np.asarray([a_trellis.elements_2_node[frozenset(element)].map_momentum for element in self.elements])
    #     #     # leaves_p = torch.from_numpy(leaves_p).float().view(1,-1,4)
    #     #     # # print("leaves momentum = ",leaves_p)
    #     #
    #     # else:
    #     self.levelLHweightFit = torch.rand(len(special_nodes), requires_grad=True, device=device)
    #     # m = torch.nn.Softmax()
    #     # self.levelLHweightFit = m(self.levelLHweightFit)
    #     a_trellis.VI_params.append({"params": self.levelLHweightFit})



    def sample_distributions(self, a_trellis, root, treeLLH, treeProb, treeProbFit,n):
        """ After we save the llh (energy) for each node in the trellis, we can sample trees following the posterior distribution for the likelihood of each of them. This follows a top down approach and can start from any subtree (of the root of the tree)"""

        # global treeProbFit

        if self.is_leaf():
            return
            # if n==3:
            #     return treeProbFit
            # else:
            #     pass


        """Choose the 1st element of the parent node as a special element.
        Get all sub-clusters that contain this element and their complement.
        As this element has to be in one of the 2 sub-clusters that are the children,
        then this way we cover all possible clusterings.
        For each node, it's "energy" is the llh of the join between the node and its complement.
        Find the max energy of all possible pairs of children for this parent node."""
        special_element = list(root.elements)[0]
        logger.debug(f"special_element = {special_element}")

        """ Get nodes containing element within current subtree"""
        special_nodes = a_trellis.get_nodes_containing_element(root, special_element)


        """Root node can't be MAP (its the parent node => can't be a children node), so remove self from special_nodes"""
        special_nodes.remove(root)

        logger.debug(f"root  = {root}")
        logger.debug(f"special nodes = {special_nodes}")

        logger.debug(f"root.levelLHweight = {root.levelLHweight}")


        """Sample a node at current level following the likelihhood of each of them"""
        # node = np.random.choice(special_nodes, 1, p=root.levelLHweight)[0]
        # node = np.random.choice(root.levelLHweight, special_nodes, 1, p=root.levelLHweight)[0]
        node_id = list(torch.utils.data.WeightedRandomSampler(root.levelLHweight, 1, replacement=True))[0]
        # print("node id = ", node_id)
        node = special_nodes[node_id]

        # print("prob=", root.levelLHweightFit[node_id])
        # print("treeProbFit = ", treeProbFit)
        # treeProb.append((root.levelLHweight,node_id))
        treeProb.append(root.levelLHweight[node_id])
        # treeProbFit *=torch.tensor([root.levelLHweightFit[node_id]])
        # treeProbFit = torch.cat((treeProbFit,torch.tensor([root.levelLHweightFit[node_id]])),0)
        # treeProbFit[n]=root.levelLHweightFit[node_id]
        treeProbFit.append((root.levelLHweightFit,node_id))


        # print("prob=",root.levelLHweightFit[node_id])
        # print("treeProb", treeProb)
        # print("treeProbFit = ", treeProbFit)

        complement = a_trellis.get_complement_node(node, root.elements)

        logger.debug(f"node = {node}")
        logger.debug(f"complement = {complement}")

        """ Get llh for the join of the pair {node,comlement} sampled"""
        # split_llh = likelihood.split_logLH(node.map_momentum,
        #                                    node.map_delta,
        #                                    complement.map_momentum,
        #                                    complement.map_delta,
        #                                    self.delta_min,
        #                                    self.lam)
        split_llh = self.get_energy_of_split(node, complement)
        treeLLH.append(split_llh)

        # self.sample_momentum = self.compute_momentum(node, complement)
        # self.sample_delta = self.compute_delta(node, complement)
        self.sample_features = self.compute_map_features(node, complement)


        n+=1
        """ Recursively repeat for the next level"""
        node.sample_distributions(a_trellis,
                               node,
                               treeLLH,treeProb,treeProbFit,n)

        complement.sample_distributions(a_trellis,
                                     complement,
                                     treeLLH,treeProb,treeProbFit,n)


