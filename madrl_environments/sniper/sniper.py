#!/usr/bin/env python
#
# File: run_sniper.py
#
# Created: Tuesday, November  1 2016 by ashe magalhaes <ashemag@stanford.edu>
#
import glob
import os
from os.path import join
from subprocess import call

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding
from matplotlib.patches import Rectangle

from madrl_environments import AbstractMAEnv
from six.moves import xrange
from utils import agent_utils
from utils.AgentLayer import AgentLayer
from utils.Controllers import RandomPolicy

from rltools.util import EzPickle

#################################################################
# Implements a Sniper Target Surveillance Problem in 2D
#################################################################

'''
In sniper environment a set of set of snipers must 'tag' (monitor) a set of targets 

Required arguments:
- map_matrix: the map on which agents interact

Optional arguments:
- Ally layer: list of snipers
-Opponent layer: list of targets
??? Do we include this -Ally controller: stationary policy of ally snipers
??? Do we include this -Ally controller: stationary policy of opponent targets 
-map_matrix: the map on which agents interact
-catchr: reward for 'tagging' a single target
-caughtr: reward for getting 'tagged' by an agent 
-train_pursuit: flag indicating if we are simulating snipers or targets
-initial_config: dictionary of form
??? -initial_config['allies']: the initial ally confidguration (matrix)
??? -initial_config['opponents']: the initial opponent confidguration (matrix)
'''
class Sniper(AbstractMAEnv, EzPickle):

	def __init__(self, map_pool, **kwargs):
		# kwargs = dictionary where you can pop key of size 1 off to define term 
		# if present, assign value and if not use default 
		EzPickle.__init__(self, map_pool, **kwargs)

		#initialize map, observation, reward   
		self.sample_maps = kwargs.pop('sample_maps', False)
		self.map_pool = map_pool
		map_matrix = map_pool[0]
		self.map_matrix = map_matrix
		xs, ys = self.map_matrix.shape
		self.xs = xs
		self.ys = ys
		self._reward_mech = kwargs.pop('reward_mech', 'global')
		self.obs_range = kwargs.pop('obs_range', 3)  # can see 3 grids around them by default
	   
		#assert self.obs_range % 2 != 0, "obs_range should be odd"
		self.obs_offset = int((self.obs_range - 1) / 2)
		self.flatten = kwargs.pop('flatten', True)

		#initalize snipers and targets  
		self.n_snipers= kwargs.pop('n_snipers', 1)
		self.n_targets = kwargs.pop('n_targets', 1)
		
		#self.agents = list of single agent entities that define how it should move given inputs 
		#helper function for creating list 
		self.snipers = agent_utils.create_agents(self.n_snipers, map_matrix, self.obs_range, flatten=self.flatten)
		self.targets = agent_utils.create_agents(self.n_targets, map_matrix, self.obs_range, flatten=self.flatten)
		self.sniper_layer = kwargs.pop('ally_layer', AgentLayer(xs, ys, self.snipers))
		self.target_layer = kwargs.pop('opponent_layer', AgentLayer(xs, ys, self.targets))
		n_act_sniper = self.sniper_layer.get_nactions(0)
		n_act_target = self.target_layer.get_nactions(0)
		self.sniper_controller = kwargs.pop('sniper_controller', RandomPolicy(n_act_sniper))
		self.target_controller = kwargs.pop('target_controller', RandomPolicy(n_act_target))
		self.term_sniper = kwargs.pop('term_sniper', 5.0)
		self.term_target = kwargs.pop('term_evade', -5.0)
		self.snipers_gone = np.array([False for i in xrange(self.n_snipers)])
		self.targets_gone = np.array([False for i in xrange(self.n_targets)])

		# initialize remainder of state  
		self.layer_norm = kwargs.pop('layer_norm', 10)
		self.n_catch = kwargs.pop('n_catch', 2)
		self.random_opponents = kwargs.pop('random_opponents', False)
		self.max_opponents = kwargs.pop('max_opponents', 10)     
		self.current_agent_layer = np.zeros((xs, ys), dtype=np.int32)
		self.catchr = kwargs.pop('catchr', 0.01)
		self.caughtr = kwargs.pop('caughtr', -0.01)
		self.urgency_reward = kwargs.pop('urgency_reward', 0.0)
		self.include_id = kwargs.pop('include_id', True)
		self.ally_actions = np.zeros(n_act_purs, dtype=np.int32)
		self.opponent_actions = np.zeros(n_act_ev, dtype=np.int32)
		self.train_sniper = kwargs.pop('train_sniper', True)
		
		'''
		NOT SURE WHATS HAPPENING HERE 
		'''
		# if self.train_pursuit:
		#     self.low = np.array([0.0 for i in xrange(3 * self.obs_range**2)])
		#     self.high = np.array([1.0 for i in xrange(3 * self.obs_range**2)])
		#     if self.include_id:
		#         self.low = np.append(self.low, 0.0)
		#         self.high = np.append(self.high, 1.0)
		#     self.action_space = spaces.Discrete(n_act_purs)
		#     if self.flatten: 
		#         self.observation_space = spaces.Box(self.low, self.high)
		#     else:
		#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, self.obs_range, self.obs_range))
		#     self.local_obs = np.zeros(
		#         (self.n_pursuers, 4, self.obs_range, self.obs_range))  # Nagents X 3 X xsize X ysize
		#     self.act_dims = [n_act_purs for i in xrange(self.n_pursuers)]
		# else:
		#     self.low = np.array([0.0 for i in xrange(3 * self.obs_range**2)])
		#     self.high = np.array([1.0 for i in xrange(3 * self.obs_range**2)])
		#     if self.include_id:
		#         np.append(self.low, 0.0)
		#         np.append(self.high, 1.0)
		#     self.action_space = spaces.Discrete(n_act_ev)
		#     if self.flatten: 
		#         self.observation_space = spaces.Box(self.low, self.high)
		#     else:
		#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, self.obs_range, self.obs_range))
		#     self.local_obs = np.zeros(
		#         (self.n_evaders, 4, self.obs_range, self.obs_range))  # Nagents X 3 X xsize X ysize
		#     self.act_dims = [n_act_purs for i in xrange(self.n_evaders)]
	   
		#more state set up 
		self.initial_config = kwargs.pop('initial_config', {})
		self.constraint_window = kwargs.pop('constraint_window', 1.0)
		self.curriculum_remove_every = kwargs.pop('curriculum_remove_every', 500)
		self.curriculum_constrain_rate = kwargs.pop('curriculum_constrain_rate', 0.0)
		self.curriculum_turn_off_shaping = kwargs.pop('curriculum_turn_off_shaping', np.inf)
		self.surround = kwargs.pop('surround', True)
		self.surround_mask = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

		#layers of state
		#layer 1: buildings
		#layer 2: snipers
		#layer 3: targets
		#layer 4: irrelevant
		#layer 5: suveillance  
		self.model_state = np.zeros((4,) + map_matrix.shape, dtype=np.float32)

	#################################################################
	# The functions below are the interface with MultiAgentSiulator # 
	#################################################################

	@property
	def agents(self):
	  return self.snipers

	@property
	def reward_mech(self):
		return self._reward_mech

	def seed(self, seed=None):
	  self.np_random, seed_ = seeding.np_random(seed)
	  return [seed_]

	def reset(self):
		#print "Check:", self.n_evaders, self.n_pursuers, self.catchr
	  self.snipers_gone.fill(False)
	  self.targets_gone.fill(False)

	  if self.random_opponents:
		if self.train_sniper:
		  self.n_targets = self.np_random.randint(1, self.max_opponents)
		else:
		  self.n_snipers = self.np_random.randint(1, self.max_opponents)
			
	  if self.sample_maps:
		self.map_matrix = self.map_pool[np.random.randint(len(self.map_pool))]

		x_window_start = np.random.uniform(0.0, 1.0-self.constraint_window)
		y_window_start = np.random.uniform(0.0, 1.0-self.constraint_window)
		xlb, xub = int(self.xs * x_window_start), int(self.xs * (x_window_start + self.constraint_window))
		ylb, yub = int(self.ys * y_window_start), int(self.ys * (y_window_start + self.constraint_window))
		constraints = [[xlb, xub], [ylb, yub]]

		self.snipers= agent_utils.create_agents(self.n_snipers, self.map_matrix,
													 self.obs_range, randinit=True,
													 constraints=constraints) 
		self.sniper_layer = AgentLayer(self.xs, self.ys, self.snipers)

		self.targets = agent_utils.create_agents(self.n_targets, self.map_matrix,
													 self.obs_range, randinit=True,
													 constraints=constraints)
		self.target_layer = AgentLayer(self.xs, self.ys, self.targets)

		self.model_state[0] = self.map_matrix
		self.model_state[1] = self.sniper_layer.get_state_matrix()
		self.model_state[2] = self.target_layer.get_state_matrix()
			
		if self.train_sniper:
		  return self.collect_obs(self.sniper_layer, self.snipers_gone)
		else:
		  return self.collect_obs(self.target_layer, self.targets_gone)

	def step(self, actions):
		"""
		Step the system forward. Actions is an iterable of action indecies.
		"""
		rewards = self.reward()

		if self.train_sniper:
			agent_layer = self.sniper_layer
			opponent_layer = self.taget_layer#sniper 
			opponent_controller = self.target_controller
			gone_flags = self.snipers_gone
		else:
			agent_layer = self.target_layer #agent layer = surveillance layer 
			opponent_layer = self.sniper_layer
			opponent_controller = self.sniper_controller
			gone_flags = self.targets_gone

		 # move allies
		if isinstance(actions, list) or isinstance(actions, np.ndarray):
		# move all agents
			for i, a in enumerate(actions):
				agent_layer.move_agent(i, a)
			else:
				# ravel it up
				act_idxs = np.unravel_index(actions, self.act_dims)
				for i, a in enumerate(act_idxs):
					agent_layer.move_agent(i, a)

		# move opponents
		for i in xrange(opponent_layer.n_agents()):
		# controller input should be an observation, but doesn't matter right now
			action = opponent_controller.act(self.model_state)
			opponent_layer.move_agent(i, action)

		# model state always has form: map, purusers, opponents, current agent id
		self.model_state[0] = self.map_matrix
		self.model_state[1] = self.sniper_layer.get_state_matrix()
		self.model_state[2] = self.target_layer.get_state_matrix()

		# remove agents that are caught
		ta_remove, sn_remove, snipers_who_remove = self.remove_agents()

		obslist = self.collect_obs(agent_layer, gone_flags)

		# add caught rewards
		rewards += self.term_sniper * snipers_who_remove
	  
		# urgency reward to speed up catching
		rewards += self.urgency_reward
		done = self.is_terminal

		if self.reward_mech == 'global':
			return obslist, [rewards.mean()] * self.n_snipers, done, {'removed': ta_remove}
	  
		return obslist, rewards, done, {'removed': ta_remove}

	def update_curriculum(self, itr):
	  self.constraint_window += self.curriculum_constrain_rate # 0 to 1 in 500 iterations
	  self.constraint_window = np.clip(self.constraint_window, 0.0, 1.0)
	  
	  # remove agents every 10 iter?
	  if itr != 0 and itr % self.curriculum_remove_every == 0 and self.n_snipers > 4:
		self.n_targets -= 1
		self.n_snipers -= 1
	  if itr > self.curriculum_turn_off_shaping:
		self.catchr = 0.0

	def render(self, plt_delay=1.0):
	  plt.matshow(self.model_state[0].T, cmap=plt.get_cmap('Greys'), fignum=1)
	  for i in xrange(self.sniper_layer.n_agents()):
		x, y = self.sniper_layer.get_position(i)
		plt.plot(x, y, "r*", markersize=12)
		if self.train_sniper:
		  ax = plt.gca()
		  ofst = self.obs_range / 2.0
		  ax.add_patch(Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range, alpha=0.5,
								  facecolor="#FF9848"))
		  for i in xrange(self.target_layer.n_agents()):
			x, y = self.target_layer.get_position(i)
			plt.plot(x, y, "b*", markersize=12)
			if not self.train_sniper:
			  ax = plt.gca()
			  ofst = self.obs_range / 2.0
			  ax.add_patch(
			  Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range, alpha=0.5,
								  facecolor="#009ACD"))
		  #plt.pause(plt_delay)
		  #plt.clf()

	def animate(self, act_fn, nsteps, file_name, rate=1.5, verbose=False):
		"""
		Save an animation to an mp4 file.
		"""
		plt.figure(0)
		# run sim loop
		o = self.reset()
		file_path = "/".join(file_name.split("/")[0:-1])
		temp_name = join(file_path, "temp_0.png")
		# generate .pngs
		self.save_image(temp_name)
		removed = 0
		for i in xrange(nsteps):
			a = act_fn(o)
			o, r, done, info = self.step(a)
			temp_name = join(file_path, "temp_" + str(i + 1) + ".png")
			self.save_image(temp_name)
			removed += info['removed']
			if verbose:
				print r, info
			if done:
				break
		if verbose: print "Total removed:", removed
		# use ffmpeg to create .pngs to .mp4 movie
		ffmpeg_cmd = "ffmpeg -framerate " + str(rate) + " -i " + join(
		file_path, "temp_%d.png") + " -c:v libx264 -pix_fmt yuv420p " + file_name
		call(ffmpeg_cmd.split())
		
		# clean-up by removing .pngs
		map(os.remove, glob.glob(join(file_path, "temp_*.png")))

	def save_image(self, file_name):
		plt.cla()
		plt.matshow(self.model_state[0].T, cmap=plt.get_cmap('Greys'), fignum=0)
		x, y = self.pursuer_layer.get_position(0)
		plt.plot(x, y, "r*", markersize=12)
		for i in xrange(self.pursuer_layer.n_agents()):
			x, y = self.pursuer_layer.get_position(i)
			plt.plot(x, y, "r*", markersize=12)
			if self.train_sniper:
				ax = plt.gca()
				ofst = self.obs_range / 2.0
				ax.add_patch(
				Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range, alpha=0.5,facecolor="#FF9848"))
		for i in xrange(self.evader_layer.n_agents()):
			x, y = self.evader_layer.get_position(i)
			plt.plot(x, y, "b*", markersize=12)
			if not self.train_sniper:
				ax = plt.gca()
				ofst = self.obs_range / 2.0
				ax.add_patch(
				Rectangle((x - ofst, y - ofst), self.obs_range, self.obs_range, alpha=0.5,facecolor="#009ACD"))

				xl, xh = -self.obs_offset - 1, self.xs + self.obs_offset + 1
		yl, yh = -self.obs_offset - 1, self.ys + self.obs_offset + 1
		plt.xlim([xl, xh])
		plt.ylim([yl, yh])
		plt.axis('off')
		plt.savefig(file_name, dpi=200)


  	def reward(self):
		"""
		Computes the joint reward for pursuers

		look at target layer and see if its adjacent 
		positive reward defaul value 

		reward: seeing targets 
		penalty: if you see ballistic threat 
		return list of rewards 

		iterate through all snipers and return list of 

		"""
		# rewarded for each tagged evader
		ps = self.sniper_layer.get_state_matrix()  # pursuer positions
		es = self.target_layer.get_state_matrix()  # evader positions

		rewards = [
				self.catchr *
		  np.sum(
			es[np.clip(self.sniper_layer.get_position(i)[0] +
				self.surround_mask[:,0], 0, self.xs-1), #surround_mask allows you to look at submatrix 
														#sum to get number of adversaries 
			  np.clip(self.sniper_layer.get_position(i)[1] +
			  self.surround_mask[:,1], 0, self.ys-1)]
			  ) 
		  for i in xrange(self.n_snipers)
		]
		return np.array(rewards)

	@property
	def is_terminal(self):
		#ev = self.evader_layer.get_state_matrix()  # evader positions
		#if np.sum(ev) == 0.0:
		if self.target_layer.n_agents() == 0:
			return True
		return False

	def update_ally_controller(self, controller):
		self.ally_controller = controller

	def update_opponent_controller(self, controller):
		self.opponent_controller = controller

#just for serializing object, do not implement for sniper 
# def __getstate__(self):
#         d = EzPickle.__getstate__(self)
#         d['constraint_window'] = self.constraint_window
#         d['n_evaders'] = self.n_evaders
#         d['n_pursuers'] = self.n_pursuers
#         d['catchr'] = self.catchr
#         return d

	def __setstate__(self, d):
		# curriculum update attributes here for parallel sampler
		EzPickle.__setstate__(self, d)
		self.constraint_window = d['constraint_window']
		self.n_targets = d['n_targets']
		self.n_snipers = d['n_snipers']
		self.catchr = d['catchr']

#################################################################

	def n_agents(self):
		return self.sniper_layer.n_agents()

	def collect_obs(self, agent_layer, gone_flags):
		obs = []
		nage = 0
		for i in xrange(self.n_agents()):
			if gone_flags[i]:
				obs.append(None)
			else:
				o = self.collect_obs_by_idx(agent_layer, nage)
				obs.append(o)
				nage += 1
		return obs

	#self.model_state = 3D matrix

	def collect_obs_by_idx(self, agent_layer, agent_idx):
		# returns a flattened array of all the observations
		n = agent_layer.n_agents()
		self.local_obs[agent_idx][0].fill(1.0/self.layer_norm)  # border walls set to -0.1?
		xp, yp = agent_layer.get_position(agent_idx)

		xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

		self.local_obs[agent_idx, 0:3, xolo:xohi, yolo:yohi] = np.abs(
													self.model_state[0:3, xlo:xhi, ylo:yhi]) / self.layer_norm
		self.local_obs[agent_idx, 3, self.obs_range/2, self.obs_range/2] = float(agent_idx) / self.n_agents()

		if self.flatten:
			o = self.local_obs[agent_idx][0:3].flatten() 
			if self.include_id:
				o = np.append(o, float(agent_idx) / self.n_agents())
			return o
		# reshape output from (C, H, W) to (H, W, C)
		#return self.local_obs[agent_idx]

		return np.rollaxis(self.local_obs[agent_idx], 0, 3)

	def obs_clip(self, x, y):
		# :( this is a mess, beter way to do the slicing? (maybe np.ix_)
		xld = x - self.obs_offset
		xhd = x + self.obs_offset
		yld = y - self.obs_offset
		yhd = y + self.obs_offset
		xlo, xhi, ylo, yhi = (np.clip(xld, 0, self.xs - 1), np.clip(xhd, 0, self.xs - 1),
							  np.clip(yld, 0, self.ys - 1), np.clip(yhd, 0, self.ys - 1))
		xolo, yolo = abs(np.clip(xld, -self.obs_offset, 0)), abs(np.clip(yld, -self.obs_offset, 0))
		xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
		return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1


	def remove_agents(self):
		"""
		Remove agents that are caught. Return tuple (n_evader_removed, n_pursuer_removed, purs_sur)
		purs_sur: bool array, which pursuers surrounded an evader
		"""
		n_sniper_removed = 0
		n_target_removed = 0
		removed_sniper = []
		removed_target= []

		ai = 0
		rems = 0
		xpur, ypur = np.nonzero(self.model_state[1])
		purs_sur = np.zeros(self.n_snipers, dtype=np.bool)
		for i in xrange(self.n_evaders):
			if self.evaders_gone[i]:
				continue
			x, y = self.target_layer.get_position(ai)
			if self.surround:
				pos_that_catch = self.surround_mask + self.target_layer.get_position(ai)
				truths = np.array(
					[np.equal([xi, yi], pos_that_catch).all(axis=1) for xi, yi in zip(xpur, ypur)])
				if np.sum(truths.any(axis=0)) == self.need_to_surround(x, y):
					removed_target.append(ai - rems)
					self.targets_gone[i] = True
					rems += 1
					tt = truths.any(axis=1)
					for j in xrange(self.n_snipers):
						xpp, ypp = self.sniper_layer.get_position(j)
						tes = np.concatenate((xpur[tt], ypur[tt])).reshape(2, len(xpur[tt]))
						tem = tes.T == np.array([xpp, ypp])
						if np.any(np.all(tem, axis=1)):
							purs_sur[j] = True
				ai += 1
			else:
				if self.model_state[1, x, y] >= self.n_catch:
					# add prob remove?
					removed_target.append(ai - rems)
					self.targets_gone[i] = True
					rems += 1
					for j in xrange(self.n_snipers):
						xpp, ypp = self.sniper_layer.get_position(j)
						if xpp == x and ypp == y:
							purs_sur[j] = True
				ai += 1

		ai = 0
		for i in xrange(self.sniper_layer.n_agents()):
			if self.snipers_gone[i]:
				continue
			x, y = self.sniper_layer.get_position(i)
			# can remove pursuers probabilitcally here?
		for ridx in removed_target:
			self.target_layer.remove_agent(ridx)
			n_target_removed += 1
		for ridx in removed_sniper:
			self.sniper_layer.remove_agent(ridx)
			n_sniper_removed += 1
		return n_target_removed, n_sniper_removed, purs_sur

	def need_to_surround(self, x, y):
		"""
			Compute the number of surrounding grid cells in x,y position that are open 
			(no wall or obstacle)
		"""
		tosur = 4
		if x == 0 or x == (self.xs - 1):
			tosur -= 1
		if y == 0 or y == (self.ys - 1):
			tosur -= 1
		neighbors = self.surround_mask + np.array([x, y])
		for n in neighbors:
			xn, yn = n
			if not 0 < xn < self.xs or not 0 < yn < self.ys:
				continue
			if self.model_state[0][xn, yn] == -1:
				tosur -= 1
		return tosur

