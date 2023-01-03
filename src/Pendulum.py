import time
import pdb
import contextlib
import os
import numpy as np

import gymnasium as gym
import torch

import Problem
import Utils

class Pendulum(Problem.Problem):
    """
    """

    obs_dims=3
    action_dims=1
    
    def __init__(self, max_time_steps=200, display=False, logdir="/tmp/"):
        
        super().__init__()

        self.env=gym.make("Pendulum-v1",render_mode="human")
        self.max_time_steps=max_time_steps
        self.display=display

        #self.logdir=logdir
        self.logdir=""

    @staticmethod
    def get_obs_dims():
        return Pendulum.obs_dims

    @staticmethod
    def get_action_dims():
        return Pendulum.action_dims

    def get_action_set(self):#is this used?
        raise Exception("not implemented")
        #return self.action_set

    def __call__(self, rnn_ag, noise_f=None):
        """
        rnn_ag should provide the interface specified in Problem.Problem

        returns 
                total_reward  float
                o_a_r         list    [[obs_0, action_0, reward_0], ..., [obs_t, action_t, reward_t]]
                                      reward_i corresponds to the transition obs_i to obs_{i+1}. 
                                      Terminal states wont be appended.
                task_solved   bool    If the agent reaches the flag, we consider it success.
                step_i        int     number of steps spent in eval
        """
        
        device=torch.device("cpu")#otherwise you have N agents on N processes that want to use 10^4000GBs on a single GPU
        rnn_ag.to(device)

        if self.logdir:
            #create logs for each process running __call__ via futures.map
            log_name=self.logdir+"Pendulum"+Utils.rand_string()+"_"+str(os.getpid())
            print("created logdir at ", log_name)
            log_fl=open(log_name,"w")
       
      
        task_solved=False
        is_done=False 
        if hasattr(rnn_ag, "eval"):
            rnn_ag.eval() 

        context_manager=torch.no_grad() if isinstance(rnn_ag, torch.nn.Module) else contextlib.suppress()

        o_a_r=[]
        extra_info=""

        with context_manager:

            #their api is not very flexbile, this is just a hack to allow the user to enable/disable display online
            self.env.env.env.env.render_mode=None

            obs_r,_=self.env.reset()
            obs=torch.Tensor(obs_r)

            total_reward=0
            for step_i in range(self.max_time_steps):

                if step_i%10==0 and self.logdir:
                    log_fl.write(str(step_i)+"   "+str(total_reward)+"   "+str(is_done)+"   "+str(c_action)+"   "+str(extra_info)+"  "+"\n")
                    log_fl.flush()

                if self.display:
                    self.env.env.env.env.render_mode="human"
                    self.env.render()

                c_action=rnn_ag(obs.unsqueeze(0).unsqueeze(0).to(device))

                action=c_action+noise_f(1) if noise_f is not None else c_action
                action[action>1]=1.0
                action[action<-1]=-1.0
                action*=2#because Pendulum expects actions in [-2.0, 2.0] and theres a tanh at the end of the policy network

                obs_np, reward, is_done, truncated, extra_info=self.env.step(action.squeeze(1).numpy())

                o_a_r.append([obs.clone().unsqueeze(0), action, reward.item()])
                obs=torch.Tensor(obs_np.copy())
                total_reward+=reward
                
                if is_done:#Pendulum is wrapped in TimeLimit, one can get the unwrapped env by seeting env=wrapped_env.env
                           #Here I'll leave the wrapped one so that we match the conditions of popular benchmarks
                    if "TimeLimit.truncated" in extra_info:#the timelimit has been reached
                        task_solved=False if extra_info["TimeLimit.truncated"] else True
                    else:#I don't think that happends with Pendulum
                        task_solved=True
                    break

        
        if self.logdir:
            log_fl.close()

        self.env.close()#for this problem too, this just closes the viewer
        return total_reward, o_a_r, task_solved, step_i 


