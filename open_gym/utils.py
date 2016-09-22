import matplotlib.pyplot as plt
from matplotlib import colors

import time, numpy, sys, PIL

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def keyWait():
    input("Press Enter to continue...")

def printFPS(step):
    now = time.clock()
    timeElapsed = now - printFPS.lastTime
    stepsDone = step - printFPS.lastStep

    eprint("FPS:", stepsDone / timeElapsed)

    printFPS.lastTime = now
    printFPS.lastStep = step

printFPS.lastTime = time.clock()
printFPS.lastStep = 0

def displayBrain(brain, res=25):    
    mapV, mapA = mapBrain(brain, res)

    plt.close()
    plt.show()  

    fig = plt.figure(figsize=(5,7))
    fig.add_subplot(211)

    plt.imshow(mapV)
    plt.colorbar(orientation='vertical')

    fig.add_subplot(212)

    cmap = colors.ListedColormap(['yellow', 'blue', 'white', 'red'])
    bounds=[-1.5,-0.5,0.5,1.5,2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(mapA, cmap=cmap, norm=norm)        
    cb = plt.colorbar(orientation='vertical', ticks=[-1,0,1,2])

    plt.pause(0.001)

def mapBrain(brain, res):
    mapV = numpy.zeros( (2 * res, 2 * res) )
    mapA = numpy.zeros( (2 * res, 2 * res) )

    for i1 in range(2 * res):
        for i2 in range(2 * res):
            s = numpy.array( [ (i1 - res) / res, (i2 - res) / res ] )
            mapV[i1, i2] = numpy.amax(brain.predictOne(s))	# TODO: more efficient to predict all at once
            mapA[i1, i2] = numpy.argmax(brain.predictOne(s))

    return (mapV, mapA)

def showImage(imgData):
    img = PIL.Image.fromarray(imgData)
    img.show()

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
