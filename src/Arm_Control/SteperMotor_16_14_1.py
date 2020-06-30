# This file will mimic the stepper motors
import Parameters as p
from time import sleep
#import RPi.GPIO as GPIO

class StepperMotor_16_14_1:
    def __init__(self,homePos_in, dirPin_in,stepPin_in,fileName):
        self.stepPos = 0
        self.CW = 1
        self.CCW = 0
        self.stepDelay = p.secPerStep
        self.dirPin = dirPin_in
        self.stepPin = stepPin_in
        self.homeStep = homePos_in
       # GPIO.setMode(GPIO.BCM) # basic set up
       # GPIO.setup(self.dirPin,GPIO.OUT) # set up pins 
       # GPIO.setup(self.stepPin,GPIO.OUT)
       # GPIO.output(self.dirPin,CW) # set direction of motor ( 1 == CW)

        #for simulation
        self.f = open(fileName, "w")
        self.publish_step()
        
       

    #function returns current position(integer steps)
    def get_pos (self):
       return (self.stepPos)
    
    #return angle from zero in degrees
    def get_angle (self):
        return(self.stepPos*p.degPerStep)
        
    def inc_step(self):
        self.stepPos +=1
        self.publish_step()
        return(self.stepPos)

    def dec_step(self):
        self.stepPos -=1
        self.publish_step()
        return(self.stepPos)

    def add_step(self,step_in):
        self.stepPos += step_in
        self.publish_step()
        return(self.stepPos)

    def publish_step(self):
        self.f.seek(0)
        self.f.truncate()
        self.f.write(str(self.stepPos))
        self.f.flush()
        
    #function returns number of steps from current position to degree argument 
    def get_steps_to_deg(self,target):
        targetStep = int(round(target/p.degPerStep))
        return(targetStep-self.get_pos())

    #function to move stepper to target location argument (degrees)
    def move_to_deg(self,deg_in):
        # get the number of steps to move 
        numSteps = self.get_steps_to_deg(deg_in)

        # move this stepper motor
        #set dircetion determinig if step is even or odd
        if numSteps <0:
            #GPIO.output(self.dirPin,self.CW)
            # move the step num
            for x in range (abs(numSteps)):
                #GPIO.ouput(self.stepPin,GPIO.HIGH)
                sleep(self.stepDelay)
                #GPIO.output(self.stepPin,GPIO.LOW)
                sleep(self.stepDelay)
                self.dec_step()

        elif numSteps >=0:
            #GPIO.output(self.dirPin,self.CCW)
            # move the step num
            for x in range (abs(numSteps)):
                #GPIO.ouput(self.stepPin,GPIO.HIGH)
                sleep(self.stepDelay)
                #GPIO.output(self.stepPin,GPIO.LOW)
                sleep(self.stepDelay)
                self.inc_step()     
        
        return(self.get_pos())

    #retun motor to home zone and reset step counter 
    def move_home(self):
        #move motors untill limit switch is pressed
        
        #set count to home position 
        self.stepPos = self.homeStep
        self.publish_step()
        
mc = StepperMotor_16_14_1(0,p.smDirPin,p.smStepPin,"smFile.txt")
