# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizers threads.
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras import backend as K

#-- constants
ENV = 'CartPole-v0'
THREADS = 4
OPTIMIZERS = 2

GAMMA = 0.99
ENTROPY = 0.01

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

NUM_ACTIONS = 2
NUM_STATE = 4

MIN_BATCH = 64
LEARNING_RATE = 2e-2

#---------
NONE_STATE = np.zeros(NUM_STATE)

class Brain:
	train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
	lock_queue = threading.Lock()

	def __init__(self):
		self.session = tf.Session()
		K.set_session(self.session)

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

	def _build_model(self):

		l_input = Input( batch_shape=(None, NUM_STATE) )
		l_dense = Dense(64, activation='relu')(l_input)

		out_actions = Dense(2, activation='softmax')(l_dense)
		out_value   = Dense(1, activation='linear')(l_dense)

		model = Model(input=l_input, output=[ out_actions, out_value ])

		return model

	def _build_graph(self, model):
		s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
		a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
		r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
		p, v = model(s_t)

		log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)					# maximize policy
		loss_value  = tf.square(advantage)										# minimize value error
		entropy = tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + ENTROPY * entropy)
		# TODO : try 0.5 * loss_value

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def _optimize(self):
		if len(self.train_queue[0]) < MIN_BATCH:
			time.sleep(0)	# yield
			return

		with self.lock_queue:
			s, a, r, s_, s_mask = self.train_queue

			self.train_queue[0] = []	#empty
			self.train_queue[1] = []
			self.train_queue[2] = []
			self.train_queue[3] = []
			self.train_queue[4] = []


		s = np.vstack(s)
		a = np.vstack(a)
		r = np.vstack(r)
		s_ = np.vstack(s_)
		s_mask = np.vstack(s_mask)

		# predict V(s')
		v = self.predict_v(s_)
		r_ = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state

		if len(s) > 2*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r_})


	def train_push(self, s, a, r, s_):
		with self.lock_queue:
			self.train_queue[0].append(s)
			self.train_queue[1].append(a)
			self.train_queue[2].append(r)

			if s_ is None:
				self.train_queue[3].append(NONE_STATE)
				self.train_queue[4].append(0.)
			else:	
				self.train_queue[3].append(s_)
				self.train_queue[4].append(1.)

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)
			return p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return v

#---------
class Agent:
	def __init__(self, eps):
		self.eps = eps
		self.memory = []	# used for n_step return
		self.R = 0.

	def act(self, s):
		if random.random() < self.eps:
			return random.randint(0, NUM_ACTIONS-1)

		else:
			s = np.array([s])
			p = brain.predict_p(s)[0]

			# a = np.argmax(p)
			a = np.random.choice(NUM_ACTIONS, p=p)

			return a

	def train(self, s, a, r, s_):
		def get_sample(memory, n):
			# compute R; there's certainly a more optimal way..
			r = 0.
			for i in range(n):
				r += memory[i][2] * (GAMMA ** i)

			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, r, s_

		a = np_utils.to_categorical([a], NUM_ACTIONS)[0]
		self.memory.append( (s, a, r, s_) )

		if s_ is None:	# terminal state
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				brain.train_push(s, a, r, s_)

				self.memory.pop(0)		

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			brain.train_push(s, a, r, s_)

			self.memory.pop(0)		
		
#---------
class Environment(threading.Thread):
	stop_signal = False

	def __init__(self, render=False, eps=0.3):
		threading.Thread.__init__(self)

		self.render = render
		self.env = gym.make(ENV)
		self.agent = Agent(eps=eps)

	def runEpisode(self):
		s = self.env.reset()

		R = 0
		while True:         
			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				s_ = None

			self.agent.train(s, a, r, s_)

			s = s_
			R += r

			if done:
				break

		print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True

#---------
class Optimizer(threading.Thread):
	stop_signal = False

	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		while not self.stop_signal:
			brain._optimize()

	def stop(self):
		self.stop_signal = True

#-- main
brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for e in envs:
	e.start()

for o in opts:
	o.start()


time.sleep(30)

for e in envs:
	e.stop()
for e in envs:
	e.join()

for o in opts:
	o.stop()
for o in opts:
	o.join()


print("Training finished")

env = Environment(render=True, eps=0.)
env.run()