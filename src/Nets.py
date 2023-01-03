import pdb
import copy

import torch
from termcolor import colored

_non_lin_dict={"tanh":torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid, "leaky_relu":torch.nn.functional.leaky_relu}

def _identity(x):
    """
    because pickle and thus scoop don't like lambdas...
    """
    return x

class MLP(torch.nn.Module):
    """
    Vanilla multilayer perceptron
    """

    def __init__(self,
            in_d,
            out_d,
            num_hidden,
            hidden_dim,
            non_lin="tanh",
            output_normalisation=""):
        
        torch.nn.Module.__init__(self)

        self.mds=torch.nn.ModuleList([torch.nn.Linear(in_d, hidden_dim)])
        
        for i in range(num_hidden-1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, out_d))

        self.non_lin=_non_lin_dict[non_lin] 

        self.output_normaliser=_non_lin_dict[output_normalisation] if output_normalisation else _identity


    def forward(self, x):
        
        out=x
        for md in self.mds[:-1]:
            out=self.non_lin(md(out))
        out=self.mds[-1](out)
        out=self.output_normaliser(out)
        return out

class StateRepresentation(torch.nn.Module):
    """
    """

    def __init__(self,
            obs_dims,
            h_dims,
            depth):
        
        torch.nn.Module.__init__(self)
    
        self.obs_dims=obs_dims
        self.h_dims=h_dims
        self.depth=depth

        self._RNN=torch.nn.ModuleList([torch.nn.GRU(input_size=obs_dims,
                hidden_size=h_dims,
                num_layers=depth,
                bias=True,
                batch_first=False,
                dropout=0)])
        
    def create_init_hidden_state(self,batch_size, init_type="zeros"):

        if init_type=="zeros":
            return torch.zeros(self.depth, batch_size, self.h_dims)
        else:
            raise Exception("not supported")

    def forward(self, x_t, h_t):
        """
        args
            x_t should be of shape L*N*in_dim
                with L sequence lenght, N batch size
            h_t Previous hidden state, should be of shape rnn_depth*N*h_dim
       
        returns 
            out_last  output from last step
            h_next    next hidden state, of shape rnn_depth*N*h_dim
        """
        self._RNN[0].flatten_parameters()#to prevent the non-contiguous memory warning 
        #out is of shape L*N*h_dim
        #h_next is the next hidden dim of shape rnn_depth*N*h_dim
        out, h_next=self._RNN[0](x_t, h_t)
        out_last=out[-1]

        return out_last, h_next



class DeterministicPolicyContinuous(torch.nn.Module):

    def __init__(self,
            repr_dim,
            action_dims):
        
        torch.nn.Module.__init__(self)

        self.repr_dim=repr_dim
        self.action_dims=action_dims
        
        self._decision=MLP(in_d=self.repr_dim,
            out_d=self.action_dims,
            num_hidden=2,
            hidden_dim=self.action_dims*2,
            non_lin="leaky_relu",
            output_normalisation="tanh")



    def forward(self, s_t):
        """
        args 
            s_t state representation, should be of dims batch_sz*state_dims
        returns 
            action   shape N*a_dim
        """
        action=self._decision(s_t)
        
        return action
    

class QfunctionApproximator(torch.nn.Module):

    def __init__(self,
            repr_dim,
            act_dim,
            act_embedding_dim):
        """
        action are embedded in a latent space of dim act_embedding_dim,
        and are then concatenated to the state representation before being
        processed.
        """
        
        torch.nn.Module.__init__(self)

        self.repr_dim=repr_dim
        self.act_dim=act_dim
        self.act_embedding_dim=act_embedding_dim
        
        self.embed_actions=MLP(in_d=self.act_dim,
            out_d=self.act_embedding_dim,
            num_hidden=1,
            hidden_dim=self.act_embedding_dim*2,
            non_lin="leaky_relu",#are we sure?
            output_normalisation="")

        self.predictor_input_dims=self.repr_dim+self.act_embedding_dim
        self._value_predictor=MLP(in_d=self.predictor_input_dims,
                out_d=1,
                num_hidden=3,
                hidden_dim=self.predictor_input_dims*2,
                non_lin="leaky_relu",#are we sure?
                output_normalisation="")


    def forward(self, a_t, h_t):
        """
        """
        embedding=self.embed_actions(a_t)
        val_pred_input=torch.cat([embedding, h_t],1)
        value=self._value_predictor(val_pred_input)

        return value


class RNNAgent:
    """
    Not meant for learning, just a useful wrapper to interact
    with a gym env
    """

    def __init__(self, srl_backend, decision_head):
   
        self.device=torch.device("cpu")#because it's for interacting in parallel with gym envs
                                       #otherwise each of the copies passed to scoop copies its weights on the GPU...

        #I'm getting clones here so that we don't have to move the original nets back to gpu once the exploration is complete
        self.srl_backend=copy.deepcopy(srl_backend).to(self.device)
        self.decision_head=copy.deepcopy(decision_head).to(self.device)
        self.srl_backend.eval()
        self.decision_head.eval()

        self.reset()

    def to(self,device):
        if device!=torch.device("cpu"):
            raise Exception("type not meant to be used with GPU")
        return None
    
    def reset(self):
        self.h_t=self.srl_backend.create_init_hidden_state(batch_size=1).to(self.device)

    def __call__(self, x_t):

        with torch.no_grad():
            h_out,self.h_t=self.srl_backend(x_t, self.h_t)
            act=self.decision_head(h_out)

        return act



if __name__=="__main__":

    
    
    device=torch.device("cpu")

    if 1:
        sr=StateRepresentation(obs_dims=5,
            h_dims=20,
            depth=3).to(device)

        dpc=DeterministicPolicyContinuous(repr_dim=sr.h_dims,
                action_dims=3).to(device)


        batch_sz=2
        seq_len=5
        xx_t=torch.rand(seq_len, batch_sz, sr.obs_dims).to(device)
        hh_init=sr.create_init_hidden_state(batch_size=batch_sz).to(device)

        hh_out,hh_next=sr(xx_t,hh_init)
        act=dpc(hh_out)

    if 1:
        sr=StateRepresentation(obs_dims=5,
                h_dims=20,
                depth=3).to(device)
        
        batch_sz=2
        seq_len=5
        hh_init=sr.create_init_hidden_state(batch_size=batch_sz).to(device)
        xx_t=torch.rand(seq_len, batch_sz, sr.obs_dims).to(device)
        hh_out,hh_next=sr(xx_t,hh_init)

        qnet=QfunctionApproximator(repr_dim=sr.h_dims,
            act_dim=8,
            act_embedding_dim=16).to(device)

        aa_t=torch.rand(batch_sz, qnet.act_dim).to(device)

        val=qnet(aa_t, hh_out)



