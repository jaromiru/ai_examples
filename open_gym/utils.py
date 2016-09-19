import matplotlib.pyplot as plt
from matplotlib import colors

import time, numpy, sys

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