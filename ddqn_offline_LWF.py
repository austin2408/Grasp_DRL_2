
import os
import wandb
import numpy as np
import torch
from tqdm import tqdm, trange

from ddqn_LWF.agent_new import Agent
from ddqn_LWF.options import Option

args = Option().create("ddqn/config/offline.yml")
agent = Agent(args)

# run = wandb.init(project="Grasp_drl")
# config = wandb.config
# config.update(args)

# crate folder
weight_path = os.path.join(args.save_folder, "weight")
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

agent.set_hdf5_memory('datasets/Logger_view.hdf5')
agent.load_pretrained_graspNet('weight/grasp_net.pth')
agent.load_pretrained_value_net('weight/value_net.pth')
# agent.load_pretrained('weight1/behavior_500_0.3912486135959625.pth')

t = trange(args.iteration)

for i in t:
    log = agent.train()
    # wandb.log(log)
    t.set_description("loss:%.3f, S/F:%.1f" %
                      (log['loss mean'], log['Success Sample Rate']))

    if (i+1) % args.save_freq == 0:
        torch.save(agent.trainer.behavior_net.state_dict(), os.path.join(
            weight_path, "behavior_%d_%.2f.pth" % (i+1, log['loss mean'])))
