# OpenGym MountainCar-v0
#
#
# actions reduced to only left & right
#
# author: Jaromir Janisch


#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  

import random, numpy, math
import gym
import utils, sys

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

def hubert_loss(y_true, y_pred):    # sqrt(1+a^2)-1
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

        # self.model.load_weights("mc.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=128, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=128, activation='relu'))

        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.0001)
        # opt = optimizers.Adadelta()

        model.compile(loss=hubert_loss, optimizer=opt)

        return model

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

    def train(self, x, y, w=None, epoch=1, verbose=0):
        self.model.fit(x, y, sample_weight=w, batch_size=32, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 0.8
MIN_EPSILON = 0.1
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def replay(self):
        ##----- debug
        if self.steps % 1000 == 0:
            P = [
                [ 0.874334,  0.703311], # s__ -> exit
                [ 0.819632,  0.69813 ], # s_ -> s__
                [ 0.765333,  0.697897], # s -> s_
                [ 0.716243,  0.109933], # s1 
                [ 0.724484,  0.10595 ], # s0 -> s1
            ]

            pred = self.brain.predict( numpy.array(P) )

            for o in pred:
                sys.stdout.write(str(o[0]) + " " + str(o[1])+" ")

            print(";")
            sys.stdout.flush()

        if self.steps % 50000 == 0:
            utils.displayBrain(self.brain, res=50)
            utils.printFPS(self.steps)

        #~~~~~~ debug

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ ([0,0] if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t            

        self.brain.train(x, y)

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

        high = self.env.observation_space.high
        low = self.env.observation_space.low

        self.mean = (high + low) / 2
        self.spread = abs(high - low) / 2

    def normalize(self, s):
        return (s - self.mean) / self.spread

    def run(self, agent):
        s = self.normalize(self.env.reset())
        R = 0 

        while True:            
            # self.env.render()

            a = agent.act(s)

            # map actions; 0 = left, 2 = right
            if a == 0: 
                a_ = 0
            elif a == 1: 
                a_ = 2

            s_, r, done, info = self.env.step(a_)
            s_ = self.normalize(s_)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            s = s_
            R += r

            agent.replay()            

            if done:
                break

        utils.eprint("Total reward:", R)

#-------------------- MAIN ----------------------------
numpy.set_printoptions(threshold=np.inf)
numpy.set_printoptions(precision=4)

PROBLEM = 'MountainCar-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = 2 #env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("mc.h5")
