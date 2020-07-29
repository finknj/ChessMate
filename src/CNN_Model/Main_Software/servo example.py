import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
servoPin = 23
GPIO.setup(servoPin, GPIO.OUT)
pwm=GPIO.PWM(servoPin, 50)

pwm.start(0)

def SetAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(servoPin, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(servoPin, False)
    pwm.ChangeDutyCycle(0)
    
SetAngle(20)
SetAngle(30)
SetAngle(40)
SetAngle(50)
SetAngle(90)
pwm.stop()
GPIO.cleanup()