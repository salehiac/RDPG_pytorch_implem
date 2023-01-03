
This is a quick and dirty implementation of the classic paper 

```
@article{heess2015memory,
  title={Memory-based control with recurrent neural networks},
  author={Heess, Nicolas and Hunt, Jonathan J and Lillicrap, Timothy P and Silver, David},
  journal={arXiv preprint arXiv:1512.04455},
  year={2015}
}
```

# Example usage

You can train an agent with

```
$cd src
$python3.8 P_RDPG.py --problem pendulum #this correponds to pendulum-v1
```

For testing, one can do something along the lines of

```
$cd src
$python3.8 P_RDPG.py --model ../well_perfoming_agents/pendulum/separated_backend/agent_A/best_model_so_far_4810_  --test 1
```

If necessary, one can sample multiple envs in parallel using scoop: 

```
$cd src
$python3.8 -m scoop -n <num_proc> P_RDPG.py <blablabla>
```

## Env visualization

You can turn visualization on/off online by setting the boolean in `online_config.json`.


