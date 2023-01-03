import subprocess
import os
import sys
from datetime import datetime
import functools
import pdb
import pickle
import random
import numpy as np
import collections
import torch
from functools import reduce
import string


class linearAnnealing:
    def __init__(self,M,m,s):
        """
        M max value (starting value)
        m min value (final value that has to be reached in s steps)
        number of steps to perform annealing in
        """
        self.M=M
        self.m=m
        self.s=s
        self.i=0

        self.range=M-m

    def step(self):

        if self.i>self.s:
            return self.m

        p=(self.i/self.s) 
        u=self.M-self.range*p
        self.i+=1

        return u

def get_current_time_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def rand_string(alpha=True, numerical=True):
    l2="0123456789" if numerical else ""
    return reduce(lambda x,y: x+y, random.choices(string.ascii_letters+l2,k=10),"")

def bash_command(cmd:list):
    """
    cmd  list [command, arg1, arg2, ...]
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    ret_code=proc.returncode

    return out, err, ret_code

def create_directory_with_pid(dir_basename,remove_if_exists=True,no_pid=False):
    while dir_basename[-1]=="/":
        dir_basename=dir_basename[:-1]
    
    dir_path=dir_basename+str(os.getpid()) if not no_pid else dir_basename
    if os.path.exists(dir_path):
        if remove_if_exists:
            bash_command(["rm",dir_path,"-rf"])
        else:
            raise Exception("directory exists but remove_if_exists is False")
    bash_command(["mkdir", dir_path])
    notif_name=dir_path+"/creation_notification.txt"
    bash_command(["touch", notif_name])
    with open(notif_name,"w") as fl:
        fl.write("created on "+get_current_time_date()+"\n")
    return dir_path


def interpolate_networks(sd1, sd2, tau):
    """
    Linear interpolation between networks: (1-tau)*sd1+tau*sd2

    sd1   state_dict of first network
    sd2   state_dict of second network
    tau   float in [0,1]
    returns new state_dict to load
    """
    assert sd1.keys()==sd2.keys(), "mismatch in network state_dict keys"
    new_sd=collections.OrderedDict()
    for k in sd1.keys():

        w1=sd1[k]
        w2=sd2[k]
        new_sd[k]=(1-tau)*w1+tau*w2

    return new_sd


def get_sum_of_model_params(mdl):

    x=[x.sum().item() for x in mdl.parameters() if x.requires_grad]

    return sum(x)

        

if __name__=="__main__":

    noise_func=linearAnnealing(10,1,s=10)
    for i in range(15):
        val=noise_func.step() 
        print(f"i=={i}, val=={val}")
