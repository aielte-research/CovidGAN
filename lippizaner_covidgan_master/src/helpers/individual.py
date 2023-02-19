import copy

import torch

from helpers.pytorch_helpers import is_cuda_enabled
from helpers.configuration_container import ConfigurationContainer


class Individual:

    def __init__(self, genome, fitness, is_local=True, learning_rate=None, optimizer_state=None,
                 source=None, id=None, iteration=0, history=None, updates_recieved=0):
        """
        :param genome: A neural network, i.e. a subclass of CompetitveNet (Discriminator or Generator)
        """
        self.genome = genome
        self.fitness = fitness
        self.is_local = is_local
        self.learning_rate = learning_rate
        self.optimizer_state = optimizer_state
        self.source = source
        self.id = id
        if history is None:
          self.history = {}
        else:
          self.history = history
        self.updates_recieved = updates_recieved


        # To keep track of which iteration the current individual is in (for logging and tracing purpose)
        self.iteration = iteration

    @staticmethod
    def decode(create_genome, params, fitness_tensor=None, is_local=True,
                    learning_rate=None, optimizer_state=None, source=None, id=None, iteration=None, history={}, updates_recieved=0):
        """
        Creates a new instance from encoded parameters and a fitness tensor
        :param params: 1d-Tensor containing all the weights for the individual
        :param fitness_tensor: 0d-Tensor containing exactly one fitness value
        :param create_genome: Function that creates either a generator or a discriminator network
        :return:
        """
        genome = create_genome(encoded_parameters=params)
        fitness = float(fitness_tensor) if fitness_tensor is not None else float('-inf')

        return Individual(genome, fitness, is_local, learning_rate, optimizer_state, source, id, iteration, history, updates_recieved)

    def clone(self):
        return Individual(self.genome.clone(), self.fitness, self.is_local, self.learning_rate,
                          copy.deepcopy(self.optimizer_state), self.source, self.id, self.iteration, self.history, self.updates_recieved)

    def add_loss(self, losses):
        self.history[self.updates_recieved] = torch.stack(losses).mean().item()
        self.updates_recieved+=1
    
    @property
    def name(self):
        """ Uniquely identify an individual (for further logging purpose) """
        return 'Iteration{}:{}:{}'.format(self.iteration, self.source, self.id)
