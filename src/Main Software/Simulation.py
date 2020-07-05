import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib import gridspec
import Parameters as p
import math


#style.use('fivethirtyeight')

# opens the file in writing mode 

fig = plt.figure()
gs = gridspec.GridSpec(1,2,width_ratios = [1,8])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[0])
plt.ion()
phi = 0.0
theta = 0.0
rack = 0.0

def animate(i):
    while( True):
        smSteps = open(p.smFile, mode = 'r').read()
        if smSteps != "":
            phi  = int(smSteps)*p.degPerStep
            break

    while(True):
        emSteps = open(p.emFile, mode = 'r').read()
        if emSteps != "":
            theta = int(emSteps)*p.degPerStep
            break

    while(True):
        line = open(p.servoFile, mode = 'r').read()
        if line != "":
            split = line.split(',')
            #print (split)
            rack = ((float(split[0])*math.pi *p.pinionRad)/180)+p.eeOffset
            mag = float(split[1])
        
            break
        
    #print (phi,theta)

    #draw Boundry
    bx = [p.boardLength/2,p.boardLength/2,(p.boardLength/2)*-1,(p.boardLength/2)*-1,p.boardLength/2]
    by = [p.boardLength + p.boardOffset,p.boardOffset,p.boardOffset,p.boardLength + p.boardOffset,p.boardLength + p.boardOffset]
    
    # line 1 points
    x1 = [0, p.ls *math.cos(math.radians(phi))]
    y1 = [0, p.ls *math.sin(math.radians(phi))]
    # line 2 points
    x2 = [x1[1],(x1[1] +p.le *math.cos(math.radians(theta+phi)))]
    y2 = [y1[1],(y1[1] +p.le *math.sin(math.radians(theta+phi)))]
    #print(x2[1],y2[1])

    #endeffector line
    xe = [0,0]
    ye =[p.pocElevation+3,p.pocElevation-rack]

    #draw magnet dot
    circle = plt.Circle((0, p.pocElevation-rack), .4, color='green', zorder=10)
    
    ax1.clear()
    ax2.clear()
    plt.setp(ax1,xlim = (-250,250),ylim = (-50,450))
    plt.setp(ax2,xlim = (-.5,.5),ylim = (-5,p.pocElevation+10))
    # plotting the boundry
    ax1.plot(bx, by, label = "Bounds")
    #plot sholder 
    ax1.plot(x1, y1, label = "Sholder")
    #plot elbow
    ax1.plot(x2, y2, label = "Elbow")
    #plot endeffector
    ax2.plot(xe,ye, label = "EE Elevation")
    if mag == 1.0:
        ax2.add_artist(circle)



ani = animation.FuncAnimation(fig, animate, interval = 10)
plt.show()
