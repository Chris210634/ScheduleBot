from gym.envs.registration import register
import gym
from envs.cluster_env import ClusterEnv
from envs.cluster_env import Job
from envs.cluster_env import Node
from envs.cluster_env import Action
import time

register(
    id='cluster-env-v0',
    entry_point='envs:ClusterEnv',
)

def fifo(env):
    if len(env.queue) == 0:
        return None
    # always schedule first job

    for job_index in range(len(env.queue)):
        num_cores_required = env.queue[job_index].num_cores
        for node_index in range(len(env.nodes)):
            if env.nodes[node_index].num_avail_cores >= num_cores_required:
                # schedule first job on first available node
                return Action(0,node_index)
    return None

def run_simulation(scheduler):
    '''
    Run simulation with scheduler function schuduler(env) -> Action.
    return loss.
    '''
    env = gym.make('cluster-env-v0')
    env.reset()

    # sort jobs in order from least time to most time
    def get_num_timesteps(job):
        return job.num_timesteps
    env.queue.sort(key=get_num_timesteps)
    
    while 1:
        #print(len(env.queue))
        #print(env)
        action = fifo(env) # scheduler returns action

        while env.take_action(action):
            # keep coming up with actions until we take an invalid action
            action = scheduler(env)
            continue
        #time.sleep(1)
        _, reward, game_over, _ = env.step()   # one step in time
        if game_over:
            break

    #print(env.cumulative_wait_time)
    return env.cumulative_wait_time

sum_loss = 0

# Number of iterations to run the simulation for
num_iters = 1

for i in range(num_iters):
    sum_loss += run_simulation(fifo)
    
print('Average loss over {} iterations: {}'.
      format(num_iters, sum_loss/num_iters))












