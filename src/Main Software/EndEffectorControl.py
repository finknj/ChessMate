#End Effector
import Parameters as p
from time import sleep
import math
#import RPi.GPIO as GPIO

class EndEffectorControl:

    def __init__ (self):
        self.servoPin = p.servoPin
        # servo pos is a degree from 0 to 180 
        self.servoPos = 0
        #set up the GPIO outputs here
        #GPIO.setmode(GPIO.BCM)
        #GPIO.setup(self.servoPin,GPIO.OUT)
        #may have to change the frequency from 50
        #self.pwm=GPIO.PWM(self.servoPin,50)
        #set up magnet output
        self.magnetVal = 0
        self.magnetPin = p.magnetPin
        #GPI0.setup(self.magnetPin,GPIO.OUT)
        #GPIO.output(self.magnetPin,0)
        #simmulaition write to file angle
        self.f = open(p.servoFile,"w")
        self.publish_ee()

    # set the servo angle based on how tall a piece is
    #arg = name of chess peice class (pawn,king,knight)
    def set_elevation(self,pieceType):
        #get the hight from parameters file
        height = p.get_piece_height(pieceType)
        #get the angle servo should approch based on height
        angle = self.calc_servo_pos(height)
        # move servo to target position
        self.move_servo(angle)
        return (self.servoPos)

    #return rack and pinion to the top
    def return_2_top(self):
        self.move_servo(0)
        return(0)

    #use equation to calc angle of rack pinion from argument distance
    def calc_servo_pos(self,height_in):
        s = p.pocElevation- height_in
        n = 180 * s
        d = math.pi * p.pinionRad
        theta = n/d
        
        return (theta)
    
    #move servo to an angle
    def move_servo(self, angle_in):
        print("Set servo angle: ",angle_in)

        #here use GPIO to write servo position
        #self.pwm.start(0)
        # may have to change to 2.5
        duty = angle_in / 18 + 2
        #GPIO.output(self.servoPin,True)
        #self.pwm.ChangeDutyCycle(duty)
        
        self.servoPos = angle_in
        #GPIO.output(self.servoPin,False)
        #self.pwm.ChangeDutyCycle(0) 
        #self.pwm.stop()
        #write new angle to file 
        self.publish_ee()
        #delay
        sleep(1)
        
        return(self.servoPos)

    def set_magnet(self,onff):
        if onff == 1:
            self.magnetVal = 1
            #GPIO.output(self.magnetPin,self.magnetVal)
            print("Activated magnet")
        else:
            self.magnetVal = 0
            #GPIO.output(self.magnetPin,self.magnetVal)
            print("Deactivated magnet")
        self.publish_ee()
        sleep(.5)
        return(self.magnetVal)
    
    def publish_ee(self):
        self.f.seek(0)
        self.f.truncate()
        str1 = str(self.servoPos)+","+str(self.magnetVal)
        self.f.write(str1)
        self.f.flush()
    

#e = EndEffectorControl()
#e.set_magnet(1)
#e.set_elevation("king")
#e.set_elevation("pawn")
#e.set_elevation("top")

