#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"   
#-------------------- BRAIN ---------------------------
from keras.models import Sequential, Model
from keras.layers import *

STATE_SIZE = 4
ACTION_SIZE = 2

class Brain:
	def __init__(self):
		self.model = Sequential()

		self.model.add(Dense(output_dim=48, input_dim=STATE_SIZE))
		self.model.add(Activation("relu"))

		self.model.add(Dense(output_dim=24))
		self.model.add(Activation("relu"))

		self.model.add(Dense(output_dim=ACTION_SIZE))
		self.model.add(Activation("linear"))

		self.model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

	def train(self, x, y):
		self.model.fit(x, y, nb_epoch=5, batch_size=64, verbose=0)

	def predict(self, s):		
		return self.model.predict(s.reshape(1,STATE_SIZE)).flatten()

#-------------------- AGENT ---------------------------
class Agent:
	brain = Brain()

	EXPERIENCE_CAPACITY = 2096
	REPLAY_BATCH = 192
	GAMMA = 0.9

	MAX_EPSILON = 1
	MIN_EPSILON = 0.05
	LAMBDA = 0.01

	EPSILON = MAX_EPSILON

	experience = []

	def observe(self, s, a, r, s_):
		self.experience.append( (s,a,r,s_) )

		if(len(self.experience) > self.EXPERIENCE_CAPACITY):
			self.experience.pop(0)

		# slowly decrease Epsilon based on our eperience
		self.EPSILON = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * math.exp(-self.LAMBDA * len(self.experience))

	def act(self, s):
		if random.random() < self.EPSILON:
			return random.randint(0,1)
		else:
			prediction = self.brain.predict(s)
			return numpy.argmax(prediction)

	def replay(self):
		size = min(len(self.experience), self.REPLAY_BATCH)
		batch = random.sample(self.experience, size)
		x = []; y = []
		for o in batch:
			s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
			
			t = self.brain.predict(s)			

			if s_ is None:
				t[a] = r
			else:					
				t[a] = r + self.GAMMA * numpy.amax(self.brain.predict(s_))

			x.append(s)
			y.append(t)

		x = np.array(x)
		y = np.array(y)

		self.brain.train(x, y)

#-------------------- MAIN ----------------------------
import random, numpy, math
import gym

agent = Agent()
env = gym.make('CartPole-v0')

episode = 0
while True:
	episode = episode + 1
	print("Running episode #"+str(episode))
	s = env.reset()

	for i in range(250):
	    env.render()

	    a = agent.act(s)
	    s_, r, done, info = env.step(a)

	    if done: # terminal state
	    	s_ = None

	    agent.observe(s, a, r, s_)
	    s = s_

	    agent.replay()

	    # input("Press Enter to continue...")

	    if done:
	    	break
