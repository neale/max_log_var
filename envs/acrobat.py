import torch

from .acrobot import AcrobotEnv

from measures import Measure
from .task import Task, RewardFunction
import numpy as np

AcrobotUpPosition = 0.0

# [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

class AcrobotUpMeasure(Measure):
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        cos1 = next_states[:, 0]
        cos2 = next_states[:, 2]
        theta1v = next_states[:, -2]
        theta2v = next_states[:, -1]
        measure1 = torch.where(cos1 <= 0.1,  100 * torch.ones_like(cos1), torch.zeros_like(cos1)).long()
        measure2 = torch.where(cos2 <= 0.1,  100 * torch.ones_like(cos2), torch.zeros_like(cos2)).long()
        measure3 = torch.where(theta1v <= 5., 100 * torch.ones_like(theta1v), torch.zeros_like(theta1v)).long()
        measure4 = torch.where(theta2v <= 5., 100 * torch.ones_like(theta2v), torch.zeros_like(theta2v)).long()
        measure = measure1 & measure2 & measure3 & measure4
        return measure.float()


class AcrobotUpRewardFunction(RewardFunction):
    def __call__(self, state, action, next_state):
        # cosine of arms are both ~1
        # theta v are both small
        cond1 = np.abs(next_state[0]) < 0.1
        cond2 = np.abs(next_state[2]) < 0.1
        cond3 = next_state[-2] < 5.
        cond4 = next_state[-1] < 5.

        if cond1 and cond2 and cond3 and cond4:
            return 100 
        else:
            return 0


class AcrobotSparseContinuousEnv(AcrobotEnv):
    @property
    def tasks(self):
        t = dict()
        t['stand_up'] = Task(measure=AcrobotUpMeasure(),
                                 reward_function=AcrobotUpRewardFunction())
        return t
