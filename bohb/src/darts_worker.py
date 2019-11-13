import time
import os, sys
import argparse

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from helper import configuration_darts

from hpbandster.core.worker import Worker

class darts_base(Worker):
    def __init__(self, eta, min_budget, max_budget, search_space,
                 nasbench_data, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainsourcepath = '/home/ANONYMOUS/ICLR19/darts_weight_sharing_analysis'
        self.path = os.path.join(self.mainsourcepath, 'optimizers/darts')
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.nasbench_data = nasbench_data
        self.search_space = search_space
        self.seed = seed

    def compute(self, config, budget, config_id, working_directory):
        return(configuration_darts(config=darts_base.complete_config(config),
                                   budget=int(budget),
                                   min_budget=int(self.min_budget),
                                   eta=self.eta,
                                   config_id=config_id,
                                   search_space=self.search_space,
                                   nasbench_data=self.nasbench_data,
                                   seed=self.seed,
                                   directory=working_directory,
                                   darts_source=self.mainsourcepath))

    @staticmethod
    def complete_config(config):
        config['init_channels'] = 16
        config['layers'] = 9
        return config

    @staticmethod
    def get_config_space():
        config_space=CS.ConfigurationSpace()

        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('arch_learning_rate',
                                                                       lower=1e-5,
                                                                       upper=1e-2,
                                                                       log=True))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('arch_weight_decay',
                                                                       lower=1e-4,
                                                                       upper=1e-1,
                                                                       log=True))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('learning_rate',
                                                                       lower=1e-3,
                                                                       upper=1,
                                                                       log=True))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('momentum',
                                                                       lower=0,
                                                                       upper=0.99,
                                                                       log=False))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('weight_decay',
                                                                       lower=1e-5,
                                                                       upper=1e-2,
                                                                       log=False))
        config_space.add_hyperparameter(CSH.UniformFloatHyperparameter('cutout_prob',
                                                                       lower=0,
                                                                       upper=1,
                                                                       log=False))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('cutout_length',
                                                                         lower=2,
                                                                         upper=32,
                                                                         log=True))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('batch_size',
                                                                         lower=16,
                                                                         upper=128,
                                                                         log=False))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('grad_clip',
                                                                         lower=1,
                                                                         upper=9,
                                                                         log=False))


        return config_space

    @classmethod
    def data_subdir(cls):
        return 'DARTS'

