import copy
import pdb
import numpy as np
import random
import functools
import json
from collections import namedtuple

from scoop import futures
import torch
from termcolor import colored

import Nets 
import Problem
import Pendulum
import Utils
import argparse

TargetUpdateParams= namedtuple('TargetUpdateParams', 'type tau')

class RDPG:
    """
    Mainly based on R-DPG, as described in

    Heess, Nicolas, et al. "Memory-based control with 
    recurrent neural networks." arXiv preprint 
    arXiv:1512.04455 (2015).
    
    """
    def __init__(self,
            SRL_net_P,
            SRL_net_Q,
            P_net,
            Q_net, 
            problem_type,
            noise_func,
            max_buffer_size,
            target_update_params,
            logdir_root="/tmp/"):
        """
        SRL_net                  torch.nn.Module       Learnable module that compresses the history of observations into a state 
        P_net                    torch.nn.Module       Learnable Deterministic policy to learn, taking as input the embeddings produced by the SRL_net
        Q_net                    torch.nn.Module       Learnable Q function approximator, using the SRL_net outputs.
        problem_type             type                  class from which problems are instanciated. Expected to provide the Problem.Problem interface.
        noise_func               functor               
        max_buffer_size          int                   maximum number of trajectories in the replay buffer
        target_update_params     TargetUpdateParams    should be a and object providint the two fields "type" and "tau".
                                                       If type is "hard", then "tau" should be an int indicating the frequency of hard updates.
                                                       if type is "soft", tau is the weight of the learned network in a soft network update (via
                                                          linear interploation between the learned and the target), and should be in [0, 1].
                                                          While soft updates converge in expectation to the learned one, they are terribly slow
                                                          and I'm not convinced that it really makes sens to interpolate networks and expect 
                                                          functionality.
        logdir           str
        """

        self.logpath=Utils.create_directory_with_pid(logdir_root+"/prdpg_",remove_if_exists=True,no_pid=False)
        print(colored("[RDPG] Created log directory at "+self.logpath, "green",attrs=["reverse"]))

        self.replay_buffer=[]

        self.SRL_net_P=SRL_net_P
        self.SRL_net_Q=SRL_net_Q
        self.Qnet=Q_net
        self.Pnet=P_net

        self._target_SP=copy.deepcopy(self.SRL_net_P)
        self._target_SQ=copy.deepcopy(self.SRL_net_Q)
        self._target_Q=copy.deepcopy(self.Qnet)
        self._target_P=copy.deepcopy(self.Pnet)
        self._target_update_params=target_update_params
        self._num_target_updates=0

        self.problem_type=problem_type
        self.noise_func=noise_func
        self.max_buffer_size=max_buffer_size
        self.display=False#set during the main loop via the json config file

        self._SPoptimiser=torch.optim.Adam(self.SRL_net_P.parameters(), lr=1e-2, weight_decay=0.0)
        self._SQoptimiser=torch.optim.Adam(self.SRL_net_Q.parameters(), lr=1e-2, weight_decay=0.0)
        self._Qoptimiser=torch.optim.Adam(self.Qnet.parameters(), lr=1e-2, weight_decay=0.0)
        self._Poptimiser=torch.optim.Adam(self.Pnet.parameters(), lr=1e-2, weight_decay=0.0)



        self.num_total_steps=0

    def _sample_episode_from_policy(self, agent, num_samples, max_steps=None, test_only=False):
        """
        Samples num_samples full trajectories from the same environment using the
        given agent and stores them in the buffer

        if test_only is True, then the trajectory is not added to the replay buffer.
        """
        
        kwargs={"display":self.display, "max_time_steps":max_steps} if max_steps is not None else{"display":self.display}
        env=self.problem_type(**kwargs)

        noise_func=self.noise_func if not test_only else None
        ret=list(futures.map(env, 
            [copy.deepcopy(agent) for _ in range(num_samples)],
            [noise_func for _ in range(num_samples)]))
      
        avg_return=0
        for i in range(len(ret)):
            avg_return+=ret[i][0]
        avg_return/=np.sum([len(x[1]) for x in ret])

        self.num_total_steps+=np.sum([len(x[1]) for x in ret])
     
        verbose=True
        if not test_only and verbose:
            print(colored(f"avg_return={avg_return}","blue",attrs=["bold","reverse"]))
            print(colored(f"num_new_steps=={np.sum([len(x[1]) for x in ret])}","blue",attrs=["bold","reverse"]))
            print(colored(f"num_steps_so_far=={self.num_total_steps}","blue",attrs=["bold","reverse"]))
            print("reminder: self.logpath==",self.logpath)
            
        if not test_only:
            for i in range(len(ret)):
                self.replay_buffer.append(ret[i][1])
            
            #manage buffer size
            if len(self.replay_buffer)>self.max_buffer_size:
                nn=len(self.replay_buffer)-self.max_buffer_size
                np.random.shuffle(self.replay_buffer)#would removing older ones be better? Not sure.
                self.replay_buffer=self.replay_buffer[nn:]

        return avg_return


    def test_model(self, num_tests, max_steps=-1, num_parallel_tests=1):

        test_res=[]
        test_agent=Nets.RNNAgent(self.SRL_net_P, self.Pnet)
        for _ in range(num_tests):
            self.online_display_on_off()
            test_res.append(self._sample_episode_from_policy(test_agent, num_parallel_tests , max_steps=None if max_steps==-1 else max_steps, test_only=True).item())
        test_score=np.mean(test_res)
         
        return test_score, test_res



    def optimise(self,
            num_iterations,
            N1,
            batch_sz,
            batch_len,#unused
            max_steps=-1,
            testing_params={"from_iter":0,"freq":10,"num_episodes":20},
            save_every_it=False):
        """
        num_iterations     int 
        N1                 int   growth rate of replay buffer, i.e. number of new trajectories sampled from the current policy at each iteration.
        batch_sz           int   batch_size used in training the networks
        batch_len          int   UNUSED
        max_steps          int   maximum number of steps to spend in an env. If -1, defaults to what the env defines itself.
        testing_params     dict  
        """

        self.Qnet.to(DEVICE)
        self.Pnet.to(DEVICE)

        avg_return_hist=[]
        avg_test_hist=[]
        best_avg_test_score=-float("inf")
        for it in range(num_iterations):

            if it>=testing_params["from_iter"]:
                if (it-testing_params["from_iter"])%testing_params["freq"]==0:
                    test_res=[]
                    test_agent=Nets.RNNAgent(self.SRL_net_P, self.Pnet)
                    for _ in range(testing_params["num_episodes"]):
                        self.online_display_on_off()
                        test_res.append(self._sample_episode_from_policy(test_agent, N1 , max_steps=None if max_steps==-1 else max_steps, test_only=True).item())
                    test_score=np.mean(test_res)
                    avg_test_hist.append(test_score)
                    np.save(self.logpath+f"/test_avg_returns",avg_test_hist)
                    if test_score>best_avg_test_score:
                        print(colored(f"======================================> New best test score: {test_score}","magenta"))
                        best_avg_test_score=test_score
                        torch.save(self.SRL_net_P,self.logpath+f"/best_model_so_far_{it}_SP.model")
                        torch.save(self.SRL_net_Q,self.logpath+f"/best_model_so_far_{it}_SQ.model")
                        torch.save(self.Qnet,self.logpath+f"/best_model_so_far_{it}_Q.model")
                        torch.save(self.Pnet,self.logpath+f"/best_model_so_far_{it}_P.model")
                        with open(self.logpath+f"/best_model_so_far_test_score_{it}.json","w") as fl:
                            json.dump({"avg_reward":float(test_score), "num_episodes_in_test":testing_params["num_episodes"]},fl)
            

            self.online_display_on_off()

            print(colored(f"optimise: it=={it}","green",attrs=["bold"]))

            agent=Nets.RNNAgent(self.SRL_net_P, self.Pnet)

            avg_return=self._sample_episode_from_policy(agent, N1 , max_steps=None if max_steps==-1 else max_steps)
            avg_return_hist.append(avg_return)
            np.save(self.logpath+f"/train_avg_returns",avg_return_hist)

            if len(self.replay_buffer)>=batch_sz:

                for _ in range(1):
                    self._sample_and_learn(batch_sz, batch_len)

                self._update_target_networks()


                self.noise_func.step()#linear annealing of the noise std
                print(colored(f"noise_std=={self.noise_func.std}","white",attrs=["reverse"]))

                if save_every_it:
                    torch.save(self.SRL_net_P,self.logpath+f"/{it}_SPmodel")
                    torch.save(self.SRL_net_Q,self.logpath+f"/{it}_SQ.model")
                    torch.save(self.Qnet,self.logpath+f"/{it}_Q.model")
                    torch.save(self.Pnet,self.logpath+f"/{it}_P.model")
            

    def _sample_and_learn(self, batch_sz, batch_len):
        """
        batch_sz     
        batch_len     UNUSED
        """

        gamma=0.9
        complete_trajs=random.sample(self.replay_buffer, k=batch_sz)

        data_lst=[]
        self._target_SQ.eval()
        self._target_SP.eval()
        self._target_Q.eval()
        self._target_P.eval()
        with torch.no_grad():
            for traj in complete_trajs:

                #pdb.set_trace()
                timesteps=len(traj)

                h_t_t_p=self._target_SP.create_init_hidden_state(batch_size=1).to(DEVICE)#batch_size=1 because we're not using batches in the traditional sense here
                h_t_t_q=self._target_SQ.create_init_hidden_state(batch_size=1).to(DEVICE)#batch_size=1 because we're not using batches in the traditional sense here

                estimated_q_values=[]
                for ts in range(timesteps):
                    x_t=traj[ts][0]
                    a_t=traj[ts][1]
                    
                    h_out_t_p, h_t_t_p=self._target_SP(x_t.unsqueeze(0).to(DEVICE),h_t_t_p.to(DEVICE))
                    h_out_t_q, h_t_t_q=self._target_SQ(x_t.unsqueeze(0).to(DEVICE),h_t_t_q.to(DEVICE))
                    
                    estimated_optimal_action=self._target_P(h_out_t_p)
                    val=self._target_Q(estimated_optimal_action, h_out_t_q)

                    estimated_q_values.append(val)

                targets=[]
                for ts in range(timesteps-1):#we can't supervise the last step, as the next step and thus the next Q(s,mu(s)) will be unknown. We could try to learn a model though

                    r_t=traj[ts][2]
                    y_t=r_t+gamma*estimated_q_values[ts+1]
                    targets.append(y_t)

                data_lst.append(torch.cat(targets,0))

        ###Training the Q ans SRL networks
        self.SRL_net_Q.train()
        self.Qnet.train()
        self._SQoptimiser.zero_grad()
        self._Qoptimiser.zero_grad()
        h_t=self.SRL_net_Q.create_init_hidden_state(batch_size=1).to(DEVICE)#batch_size=1 because we're not using batches in the traditional sense here
        predicted_Qvals_batch_lst=[]

        for traj in complete_trajs:

            vals=[]
            for ts in range(timesteps-1):#we can'st supervise the last one
                x_t=traj[ts][0]
                a_t=traj[ts][1]

                h_out, h_t=self.SRL_net_Q(x_t.unsqueeze(0).to(DEVICE),h_t.to(DEVICE))
                vl=self.Qnet(a_t.to(DEVICE),h_out.to(DEVICE))
                vals.append(vl)
            
            predicted_Qvals_batch_lst.append(torch.cat(vals,0))
        
        
        uu=predicted_Qvals_batch_lst
        print(colored(f"predicted_Qvals_batch_lst[0]_min=={uu[0].min()}","red",attrs=["bold"]))
        print(colored(f"data_lst[0]_min=={data_lst[0].min()}","red",attrs=["bold"]))
        #pdb.set_trace()
        Q_losses=torch.cat([(data_lst[ii] - predicted_Qvals_batch_lst[ii])**2 for ii in range(len(complete_trajs))],0)
        Q_loss=Q_losses.sum()/Q_losses.shape[0]
        print("Qloss==",Q_loss)
        
        Q_loss.backward()
        self._SQoptimiser.step() 
        self._Qoptimiser.step() 
       
        ###Training the P network as well as the SRL again
        #we stick to the trajectories instead of following the policy to remain where the Q network has been trained 
        self.Qnet.eval()
        self.SRL_net_Q.eval()
        self.Pnet.train()
        self.SRL_net_P.train()
        self._SPoptimiser.zero_grad()
        self._Poptimiser.zero_grad()

        Ploss=0
        num_terms=0
        for traj in complete_trajs:

            timesteps=len(traj)#timesteps might not be the same for all trajs

            hh_p=self.SRL_net_P.create_init_hidden_state(batch_size=1).to(DEVICE)
            hh_q=self.SRL_net_Q.create_init_hidden_state(batch_size=1).to(DEVICE)
            for ts in range(timesteps):
                x_t=traj[ts][0]

                hh_out_p, hh_p=self.SRL_net_P(x_t.unsqueeze(0).to(DEVICE), hh_p)
                hh_out_q, hh_q=self.SRL_net_Q(x_t.unsqueeze(0).to(DEVICE), hh_q)
                
                suggested_act=self.Pnet(hh_out_p)
                Qval=self.Qnet(suggested_act, hh_out_q)
                Ploss=Ploss+Qval
                num_terms+=1

        Ploss=Ploss/num_terms
        print("Ploss==",Ploss)
        print("num_terms==",num_terms)
        Ploss=-1*Ploss # we want to maximise it
        Ploss.backward()
        self._SPoptimiser.step()
        self._Poptimiser.step()


        self.SRL_net_Q.eval()
        self._target_SQ.eval()
        self.Qnet.eval()
        self._target_Q.eval()

        self.SRL_net_P.eval()
        self._target_SP.eval()
        self.Pnet.eval()
        self._target_P.eval()

    def _update_target_networks(self):


        if self._target_update_params.type=="soft":
            
            print(colored("(soft) updating target networks","magenta",attrs=["bold","reverse"]))

            tau=self._target_update_params.tau
        
            nsSQ=Utils.interpolate_networks(self._target_SQ.state_dict(),
                    self.SRL_net_Q.state_dict(), 
                    tau=tau)
            
            nsSP=Utils.interpolate_networks(self._target_SP.state_dict(),
                    self.SRL_net_P.state_dict(), 
                    tau=tau)

            nsQ=Utils.interpolate_networks(self._target_Q.state_dict(),
                    self.Qnet.state_dict(), 
                    tau=tau)

            nsP=Utils.interpolate_networks(self._target_P.state_dict(),
                    self.Pnet.state_dict(), 
                    tau=tau)

            self._target_SQ.load_state_dict(nsSQ)
            self._target_SP.load_state_dict(nsSP)
            self._target_Q.load_state_dict(nsQ)
            self._target_P.load_state_dict(nsP)

            self._target_SQ.to(DEVICE)
            self._target_SP.to(DEVICE)
            self._target_P.to(DEVICE)
            self._target_Q.to(DEVICE)
            
            self._num_target_updates+=1

        elif self._target_update_params.type=="hard":

            update_freq=self._target_update_params.tau
            if self._num_target_updates%update_freq==0:
            
                print(colored("(hard) updating target networks","magenta",attrs=["bold","reverse"]))

                self._target_SQ=copy.deepcopy(self.SRL_net_Q)
                self._target_SP=copy.deepcopy(self.SRL_net_P)
                self._target_Q=copy.deepcopy(self.Qnet)
                self._target_P=copy.deepcopy(self.Pnet)

            self._num_target_updates+=1


        else:
            raise Exception("unsupported target update option")

    def online_display_on_off(self):
        with open("online_config.json","r") as live_cfg_fl:
            live_cfg=json.load(live_cfg_fl)
            self.display=live_cfg["TURN_ON_DISPLAY"]




class noise_func_annealed_std:
    """
    Uses linear annealing of the std deviation
    """
    def __init__(self,num_steps,std_max,std_min):
        self.std=std_max
        self.linann=Utils.linearAnnealing(std_max,std_min,num_steps)
    def __call__(self,n):
        return torch.randn(n)*self.std
    def step(self):
        self.std=self.linann.step()

class noise_func_cst_std:
    def __init__(self,std):
        self.std=std
    def __call__(self,n):
        return torch.randn(n)*self.std
    def step(self):
        return None

class noise_func_epsilon_greedy:
    def __init__(self,std,epsilon):
        self.std=std
        self.epsilon=epsilon
    def __call__(self,n):
        if np.random.rand() <self.epsilon:
            return torch.randn(n)*self.std
        else:
            return 0
    def step(self):
        return None



def main(test_mode,
        problem_type,
        model_basename,
        N1):
    """
    N1 is the number of episodes sampled during each exploration
    """

    if problem_type=="pendulum":

        problem_type=Pendulum.Pendulum

        if not test_mode:
            srl_net_p=Nets.StateRepresentation(obs_dims=problem_type.get_obs_dims(),
                    h_dims=12,
                    depth=2).to(DEVICE)
            
            srl_net_q=Nets.StateRepresentation(obs_dims=problem_type.get_obs_dims(),
                    h_dims=12,
                    depth=2).to(DEVICE)



            dpc=Nets.DeterministicPolicyContinuous(repr_dim=srl_net_p.h_dims,
                    action_dims=problem_type.get_action_dims()).to(DEVICE)

            qnet=Nets.QfunctionApproximator(repr_dim=srl_net_q.h_dims,
                act_dim=problem_type.get_action_dims(),
                act_embedding_dim=problem_type.get_action_dims()*4).to(DEVICE)
        else:
            srl_net_p=torch.load(model_basename+"SP.model")
            srl_net_q=torch.load(model_basename+"SQ.model")
            dpc=torch.load(model_basename+"P.model")
            qnet=torch.load(model_basename+"Q.model")
        
        soft_update=TargetUpdateParams("soft", 0.001)
        hard_update=TargetUpdateParams("hard", 20)#hard_update.tau is interpreted as the frequency at which the targets are updated via cloning

        num_iterations=5000
        pdprg=RDPG(SRL_net_P=srl_net_p,
                SRL_net_Q=srl_net_q,
                P_net=dpc,
                Q_net=qnet,
                problem_type=problem_type,
                noise_func=noise_func_annealed_std(num_steps=100,std_max=1,std_min=0.1),
                max_buffer_size=1000,
                target_update_params=hard_update,
                logdir_root="../logs/")


        print(srl_net_p)
        print(srl_net_q)
        print(dpc)
        print(qnet)

        if not test_mode:
            pdprg.optimise(num_iterations=num_iterations, 
                    N1=N1,#number of episodes sampled during each exploration
                    batch_len=-1,#unused, see documention in other parts of this file
                    #testing_params={"from_iter":num_iterations/10,"freq":10,"num_episodes":30},
                    testing_params={"from_iter":num_iterations/10,"freq":10,"num_episodes":30},
                    batch_sz=2)#number of trajectories samples for each update of the networks
        else:
            avg_test_res, test_res= pdprg.test_model(num_tests=50, num_parallel_tests=1)
            import matplotlib.pyplot as plt
            plt.hist(test_res);
            plt.show()
            return test_res

    else:

        raise Exception("unknown problem type")

    
if __name__=="__main__":

    #if torch.cuda.is_available():
    if 0:#seems slower on GPU, for reasons that are more or less obvious
        DEVICE=torch.device("cuda:0")
    else:
        DEVICE=torch.device("cpu")

    parser = argparse.ArgumentParser(description='RDPG implementation.')
    parser.add_argument('--N1', type=int,  help="number of episodes to sample for each update", default=1)
    parser.add_argument('--model', type=str,  help="For testing or resuming. [your_basename], so that your_basenameS.model, your_basenameQ.model, your_basenameP.model are loaded.", default="")
    parser.add_argument('--test', type=bool,  help="only test model", default=False)
    parser.add_argument('--problem', type=str,  help="", default="pendulum")
    
    args = parser.parse_args()
    
    assert (not args.test) or args.model, "Are you sure you want to test a random model?"

    main(args.test, args.problem, args.model, args.N1)
