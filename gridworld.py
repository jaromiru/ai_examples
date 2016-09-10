# Sample implementation of Gridworld and Reinforcement Learning with Q function and experience replay (model learnign & planning)

import numpy, os, time, copy, random

class Environment:
	walls = [	# 1 for wall; 9 for reset
				[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
				[1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,1,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1],
				[1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,9,1],
				[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
			]

	rewards = [
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
				[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			]
	
	staticReward = -0.01 # reward given on every transition
	
	startState = [19, 1]
	agentState = startState		
	

	def processAction(self, action):
		newState = copy.copy(self.agentState)

		if action == 0: #UP
			newState[0] = newState[0] - 1
		if action == 1: #RIGHT
			newState[1] = newState[1] + 1
		if action == 2: #DOWN
			newState[0] = newState[0] + 1
		if action == 3: #LEFT
			newState[1] = newState[1] - 1

		reward = self.rewards[newState[0]][newState[1]] + self.staticReward

		if self.walls[newState[0]][newState[1]] == 0:
			self.agentState = newState

		if self.walls[newState[0]][newState[1]] == 9:
			self.agentState = self.startState

		return reward

	def printMap(self):
		state = copy.deepcopy(self.walls)
		state[self.agentState[0]][self.agentState[1]] = 2

		os.system('clear')
		print(numpy.matrix(state))

class Experience:
	def __init__(self, oldState, action, newState, reward):
		self.oldState = oldState
		self.action = action
		self.newState = newState
		self.reward = reward

class Agent:
	Q = numpy.zeros((20, 20, 4))	
	exp = { }
	epsilon = 0.1

	alpha = 0.5
	gamma = 0.9

	def greedyAction(self, state):
		return numpy.argmax(self.Q[state[0]][state[1]])	# possible improvement: break the ties!

	def replayExperience(self):
		if len(self.exp) <= 0:
			return

		i = random.randint(0, len(self.exp)-1)
		e = list(self.exp.values())[i]

		self.learn(e.oldState, e.action, e.newState, e.reward)


	def addExperience(self, oldState, action, newState, reward):
		key = str(oldState[0]) + "_" + str(oldState[1]) + "_" + str(action)
		self.exp[key] = Experience(oldState, action, newState, reward)


	def decide(self, state):	
		if random.random() < self.epsilon:
			return random.randint(0,3)	#explore	
		else:
			return self.greedyAction(state) #do greedy

	def learn(self, oldState, action, newState, reward):		
		Qnew = self.Q[newState[0]][newState[1]]
		Qold = self.Q[oldState[0]][oldState[1]]
		Qold[action] = Qold[action] + self.alpha * ( reward + self.gamma * numpy.amax(Qnew) - Qold[action] )

env = Environment()
agent = Agent()

while True:
	env.printMap()

	oldState = env.agentState

	action = agent.decide(env.agentState)
	reward = env.processAction(action)

	print(agent.Q[oldState[0]][oldState[1]])
	print(action, reward)

	newState = env.agentState
	agent.learn(oldState, action, newState, reward)
	agent.addExperience(oldState, action, newState, reward)

	for i in range(1000):
		agent.replayExperience()

	# input("Press Enter to continue...")
	# time.sleep(0.01)