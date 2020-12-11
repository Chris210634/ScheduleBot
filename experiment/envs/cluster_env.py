#!/usr/bin/env python
# Core Library
import logging.config
import math
import random
from typing import Any, Dict, List, Tuple

# Third party
import gym
import numpy as np
import pkg_resources
from gym import spaces

class Action:
    def __init__(self, job_index, node_index):
        self.job_index = job_index
        self.node_index = node_index
    def __str__(self):
        return str((self.job_index, self.node_index))
        
class Job:
    def __init__(self, num_cores, num_timesteps, id, priority=0):
        self.num_cores = num_cores
        self.num_timesteps = num_timesteps
        self.id = id
        self.priority = priority

    def __str__(self):
        return str((self.num_cores, self.num_timesteps, self.id, self.priority))
        
class Node:
    def __init__(self, num_cores):
        self.num_cores = num_cores # total number of cores
        self.num_avail_cores = num_cores # current available cores

        # keep track of how long cores are occupied for.
        # (for illustration purposes only, in reality we don't really
        # care which cores are occupied, because they're all the same
        self.core_status = [0] * num_cores
        self.core_job_id = [0] * num_cores

    def schedule(self, job: Job) -> bool:
        """
        Schedule a job on this node.
        Return True if there are enough resources to schedule this job immediately.
        Return False of there are not enough resources.
        Don't schedule a job unless it can be exceuted immediately.
        (This way, we don't have to keep track of per-node queue).
        """
        if self.num_avail_cores < job.num_cores:
            return False

        # Find the available cores
        num_cores_found = 0

        for i in range(self.num_cores):
            if self.core_status[i] == 0:
                # available

                self.core_status[i] = job.num_timesteps
                self.core_job_id[i] = job.id
                
                self.num_avail_cores -= 1
                num_cores_found += 1
                if num_cores_found >= job.num_cores:
                    # found all the cores needed, we're done
                    break
 
        return True

    def step(self):
        """ Take a step in time. Return list of job IDs finished."""
        finished_jobs = []
        
        for i in range(self.num_cores):
            if self.core_status[i] != 0:
                # core busy
                self.core_status[i] -= 1
                if self.core_status[i] == 0:
                    # no longer busy
                    self.num_avail_cores += 1
                    job_id = self.core_job_id[i]
                    self.core_job_id[i] = 0
                    if not job_id in finished_jobs:
                        finished_jobs.append(job_id)
            
        return finished_jobs

    def __str__(self):
        return 'Number of cores: ' + str(self.num_cores) + '\n' +\
               'Number available: ' + str(self.num_avail_cores) + '\n' +\
               'status: ' + \
               str(self.core_status) + '\n' + \
               'jobs: ' + str(self.core_job_id)

class ClusterEnv(gym.Env):
    """
    Define a simple Banana environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self) -> None:
        self.number_of_nodes = 1
        self.number_of_cores_per_node = 8
        self.nodes = []
        for _ in range(self.number_of_nodes):
            self.nodes.append(Node(self.number_of_cores_per_node))
        self.queue = []
        self.next_job_id = 1

        # Maybe we want to minimize this. Use as loss?
        self.cumulative_wait_time = 0

        # If off-line training, generate list of jobs up-front
        self.queue = self._generate_off_line_jobs()

    def _generate_off_line_jobs(self):
        """
        For off-line scheduling, all jobs arrive at t=0.
        game continues until all jobs are scheduled.
        This function populates the queue at t=0
        """
        random.seed(564) #TUNE
        total_number_of_jobs = 16 #TUNE
        max_time = 16 #TUNE
        queue = []

        for i in range(1, total_number_of_jobs+1):
            num_cores = random.randrange(1, self.number_of_cores_per_node + 1)
            num_time = random.randrange(1, max_time + 1)
            queue.append(Job(num_cores, num_time, i))
        return queue

    def get_job_arrivals(self):
        """
        Return a set of jobs to arrive at timestep.
        If running in off-line mode (no job arrivals after t=0),
        then just return empty list.
        """
        return []
##        r = random.random()
##        capacity = self.number_of_nodes * self.number_of_cores_per_node
##        arrival_frequency = 2
##        number_jobs = 3
##        if r > 1.0 / arrival_frequency:
##            #capacity = arrival_frequency * capacity
##            # total time*cores = capacity
##            #per_job_capacity = capacity / number_jobs
##            
##            self.next_job_id += number_jobs
##
##            #n_cores = random.randrange(1,self.number_of_cores_per_node+1)
##
##            
##            # try deterministic blocks first.
##            # all add up to 32
##            return [Job(2, 8, self.next_job_id),
##                    Job(1, 8, self.next_job_id+1),
##                    Job(4, 2, self.next_job_id+2)]
##        else:
##            return []

    def step(self):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # Take action
        # self._take_action(action)
        # moved outside

        # take timestep
        total_finished_jobs = 0
        for node in self.nodes:
            finished_jobs = node.step()
            total_finished_jobs += len(finished_jobs)
        self.cumulative_wait_time += len(self.queue)

        # Calculate reward
        reward = total_finished_jobs
        
        # Get new arrivals
        new_jobs = self.get_job_arrivals()
        for job in new_jobs:
            self.queue.append(job)

        #increment priority
        for job in self.queue:
            job.priority += 1

        # Game over when queue is empty
        if len(self.queue) == 0:
            game_over = True
        else:
            game_over = False

        # Just return reward for now, because our system is fully transparent
        # i.e., the Agent has perfect knowledge of environment state.
        # return ob, reward, episode_over, debug_info
        return None, reward, game_over, {}

    def take_action(self, action):
        """
        Take action. Rreturn whether or not action was successfully taken.
        """
        if action is None:
            return False
        
        try:
            job = self.queue[action.job_index]
            if self.nodes[action.node_index].schedule(job):
                # only remove from queue if scheduled successfully
                self.queue.pop(action.job_index)
                return True
            return False
        except:
            return False # either job index out of range or node index out of range
        
    def reset(self) -> List[int]:
        """
        Reset the state of the environment and returns an initial observation.
        Returns
            -------
        observation: List[int]
            The initial observation of the space.
        """
        pass

    def _render(self, mode: str = "human", close: bool = False) -> None:
        return None

    def __str__(self):
        return_str = 'Length of queue: ' + str(len(self.queue))
##        return_str = 'Queue: \n'
##        for job in self.queue:
##            return_str += str(job)
##            return_str += '\n'
        for node in self.nodes:
            return_str += str(node)
            return_str += '\n'
        return_str += 'loss: ' + str(self.cumulative_wait_time)
        return return_str

    def _get_state(self) -> List[int]:
        """Get the observation."""
        pass







