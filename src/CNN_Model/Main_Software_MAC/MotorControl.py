from SteperMotor_16_14_1 import StepperMotor_16_14_1
import Parameters as p
import math
from threading import Thread
import time
from time import sleep


class MotorControl:
    def __init__(self):
        self.sholderMotor = StepperMotor_16_14_1(p.sHomeSteps,p.smHomePin, 1, p.smDirPin,p.smStepPin,p.smFile)
        self.elbowMotor = StepperMotor_16_14_1(p.eHomeSteps,p.emHomePin,0,p.emDirPin,p.emStepPin,p.emFile)

    # returns current motor position in degrees 
    def get_joint_deg(self):
        return([self.sholderMotor.get_angle(),self.elbowMotor.get_angle()])

    #return x and y coordiate of endeffector position
    def get_arm_coord(self):
        angles = self.get_joint_deg()
        x = (p.ls*math.cos(math.radians(angles[0]))) + (p.le*math.cos(math.radians(angles[0]+angles[1])))
        y = (p.ls*math.sin(math.radians(angles[0]))) + (p.le*math.sin(math.radians(angles[0]+angles[1])))
        return([x,y])
    
    # function returns angles (degrees) of motors to reach (x,y)
    def calc_motor_angle(self, x_in, y_in):
        ##print ("Destination: ",(x_in,y_in))

        # calculate the distance from the origin to the end point 
        D = self.distance_two_points(0,0,x_in,y_in)
        ##print ("Distance: ", D)

        # if The distance of move is larger than arm length, error 
        if (D > p.le+ p.ls):
            print ("Error: Desitnation is out of arm reach")
            return([])

        #calculate the angles the line segments should be in accordiance to positve x axis 
        phi_out = math.atan2(y_in,x_in)- (math.acos(((x_in**2)+(y_in**2)+(p.ls**2)-(p.le**2))/((2*p.ls)*D)))
        theta_out = math.acos(((x_in**2)+(y_in**2)-(p.ls**2)-(p.le**2))/((2*p.ls*p.le)))
        ##print ("Sholder joint angle: ", math.degrees(phi_out))
        ##print ("Elbow joint angle: ",math.degrees(theta_out))
        
        return ([math.degrees(phi_out), math.degrees(theta_out),D])

    #calculate distance between two points 
    def distance_two_points(self,x1,y1,x2,y2):
        #return the distance from two points calculation 
        return (math.sqrt((pow((y2-y1),2)+ pow((x2-x1),2))))

    #returns a list of points along a linear path to the destination argument 
    def calc_path_plan(self,x_in,y_in):
        #get current pos of x and y coordanates
        currentPos = self.get_arm_coord()
        #calculate parse count based on pathres 
        parseX = (x_in-currentPos[0])/p.pathRes
        parseY = (y_in-currentPos[1])/p.pathRes

        # create a linear path of points for arm to move
        # adding features like staying in a boundry or going around
        # the center axis could be added here 
        path = []
        for i in range(1,p.pathRes):
            #if x path is not in bounds, make it bounds 
            xPath = currentPos[0]+(i*parseX)
            if xPath < p.xBounds[0]:
                xPath = p.xBounds[0]
            if xPath > p.xBounds[1]:
                xPath = p.xBounds[1]
            #if y path is not in bounds, make it bounds 
            yPath = currentPos[1]+(i*parseY)
            if yPath < p.yBounds[0]:
                yPath = p.yBounds[0]
            if yPath > p.yBounds[1]:
                yPath = p.yBounds[1]
                
            path.append([xPath,yPath])

        if x_in < p.xBounds[0]:
            xPath = p.xBounds[0]
        elif x_in > p.xBounds[1]:
            xPath = p.xBounds[1]
        else:
            xPath = x_in

        if y_in < p.yBounds[0]:
            yPath = p.yBounds[0]
        elif y_in > p.yBounds[1]:
            yPath = p.yBounds[1]
        else:
            yPath = y_in
                
        path.append([xPath,yPath])
        
        #for i in range(len(path)):
        #    print(path[i])
        return(path)

    #preform all the required steps to move the robot arm to a point
    def move_robot_arm(self, x_in,y_in):
        ts1 = time.time()
        print ("Moving to destination: ",(x_in,y_in))
        # find path to target
        cX,cY = self.get_arm_coord()
        if ((x_in - cX) < 30):
            x_new = x_in-5
            print("offset X added: ", x_new)
        else:
            x_new = x_in
        
        if ((y_in - cY) > 30):
            y_new = y_in+10
            print("offset Y added: ", y_new)
        else:
            y_new = y_in
            
        path = self.calc_path_plan(x_new,y_new)
        #print(path)
        #for every point in path, move to that point
        
        for i in range(len(path)):
            angles = self.calc_motor_angle(path[i][0],path[i][1])
            if (angles == []):
                print("Error while navigating path")
                return(-1)
            # start 2 threads and wait for them to end
            # this moves the motors conjointly 
            t1 = Thread(target = self.sholderMotor.move_to_deg, args = [angles[0]])
            t2 = Thread(target = self.elbowMotor.move_to_deg, args = [angles[1]])
                        
            t1.start()
            t2.start()
            #self.sholderMotor.move_to_deg(angles[0])
            #self.elbowMotor.move_to_deg(angles[1])
            t1.join()
            t2.join()

        print("Move Finished", angles)
        ts2 = time.time()
        moveTime = float (ts2 - ts1)
        print("Move time: ",moveTime)
        return(1)

    # return arm motors to home position and set step count to home value
    def move_arm_home(self):
        t1 = Thread(target = self.sholderMotor.move_home, args = [])
        t2 = Thread(target = self.elbowMotor.move_home, args = [])

        t1.start()
        t2.start()
        #self.sholderMotor.move_home()
        #self.elbowMotor.move_home()
        t1.join()
        t2.join()
        print("Arm homed to: ",self.get_arm_coord())
        self.sholderMotor.move_to_deg(0)
        sleep(.5)
        return(0)
        
    
# mc = MotorControl()
# mc.move_arm_home()
# mc.move_robot_arm(-133.35 , 361.95)
# mc.move_robot_arm(100,300)
# mc.move_robot_arm(-100,100)
# mc.move_robot_arm(200,250)
