# This file Contains Parameters of Robot Control

#Board Parameters 
#board Length (mm)
boardLength = 304.8
#Distance from origin to board (mm)
boardOffset = 50.8

#Arm Structure 
# length of sholder segment (mm)
ls = 221.793
#length of elbow segment (mm)
le = 221.793


#stepper motor
#value of steps for home position
sHomeSteps = 0
eHomeSteps = 0
#degrees per step
degPerStep =  0.1310997815
#steps per rotation
stepsPerRotation = 2746
# seconds per step
secPerStep = .0039727207
#Sholder motor pins
smDirPin = 19 # this is the direction pin
smStepPin = 21 # signal to step pin
#elbow motor pins 
emDirPin = 22 # this is the direction pin
emStepPin = 23 # signal to step pin
smFile = "smfile.txt"
emFile = "emfile.txt"


#servo motor and end effector
servoPin = 32
magnetPin = 16
servoFile = "servofile.txt"
pinionRad = 41.38
pocElevation = 169
# end effector offset in elevation
eeOffset = 0.0


#Path Parameters
pathRes = 100


# function to return chess piece height
def get_piece_height(pieceType):
    switcher = {
        "pawn": 43,
        "knight":12,
        "rooke":13,
        "bishop":14,
        "queen":15,
        "king":80,
        "top":pocElevation
        }
    return(switcher[pieceType])

#return the coordinate system X,Y location of a square
def get_square_coord(loc_in):
    switcher = {
        1 :[ -133.35 , 336.55 ],
        2 :[ -95.25 , 336.55 ],
        3 :[ -57.14999999999999 , 336.55 ],
        4 :[ -19.049999999999983 , 336.55 ],
        5 :[ 19.05000000000001 , 336.55 ],
        6 :[ 57.150000000000006 , 336.55 ],
        7 :[ 95.25000000000003 , 336.55 ],
        8 :[ 133.35 , 336.55 ],
        9 :[ -133.35 , 298.45 ],
        10 :[ -95.25 , 298.45 ],
        11 :[ -57.14999999999999 , 298.45 ],
        12 :[ -19.049999999999983 , 298.45 ],
        13 :[ 19.05000000000001 , 298.45 ],
        14 :[ 57.150000000000006 , 298.45 ],
        15 :[ 95.25000000000003 , 298.45 ],
        16 :[ 133.35 , 298.45 ],
        17 :[ -133.35 , 260.35 ],
        18 :[ -95.25 , 260.35 ],
        19 :[ -57.14999999999999 , 260.35 ],
        20 :[ -19.049999999999983 , 260.35 ],
        21 :[ 19.05000000000001 , 260.35 ],
        22 :[ 57.150000000000006 , 260.35 ],
        23 :[ 95.25000000000003 , 260.35 ],
        24 :[ 133.35 , 260.35 ],
        25 :[ -133.35 , 222.25 ],
        26 :[ -95.25 , 222.25 ],
        27 :[ -57.14999999999999 , 222.25 ],
        28 :[ -19.049999999999983 , 222.25 ],
        29 :[ 19.05000000000001 , 222.25 ],
        30 :[ 57.150000000000006 , 222.25 ],
        31 :[ 95.25000000000003 , 222.25 ],
        32 :[ 133.35 , 222.25 ],
        33 :[ -133.35 , 184.15 ],
        34 :[ -95.25 , 184.15 ],
        35 :[ -57.14999999999999 , 184.15 ],
        36 :[ -19.049999999999983 , 184.15 ],
        37 :[ 19.05000000000001 , 184.15 ],
        38 :[ 57.150000000000006 , 184.15 ],
        39 :[ 95.25000000000003 , 184.15 ],
        40 :[ 133.35 , 184.15 ],
        41 :[ -133.35 , 146.05 ],
        42 :[ -95.25 , 146.05 ],
        43 :[ -57.14999999999999 , 146.05 ],
        44 :[ -19.049999999999983 , 146.05 ],
        45 :[ 19.05000000000001 , 146.05 ],
        46 :[ 57.150000000000006 , 146.05 ],
        47 :[ 95.25000000000003 , 146.05 ],
        48 :[ 133.35 , 146.05 ],
        49 :[ -133.35 , 107.94999999999999 ],
        50 :[ -95.25 , 107.94999999999999 ],
        51 :[ -57.14999999999999 , 107.94999999999999 ],
        52 :[ -19.049999999999983 , 107.94999999999999 ],
        53 :[ 19.05000000000001 , 107.94999999999999 ],
        54 :[ 57.150000000000006 , 107.94999999999999 ],
        55 :[ 95.25000000000003 , 107.94999999999999 ],
        56 :[ 133.35 , 107.94999999999999 ],
        57 :[ -133.35 , 69.85000000000002 ],
        58 :[ -95.25 , 69.85000000000002 ],
        59 :[ -57.14999999999999 , 69.85000000000002 ],
        60 :[ -19.049999999999983 , 69.85000000000002 ],
        61 :[ 19.05000000000001 , 69.85000000000002 ],
        62 :[ 57.150000000000006 , 69.85000000000002 ],
        63 :[ 95.25000000000003 , 69.85000000000002 ],
        64 :[ 133.35 , 69.85000000000002 ]
        }
    return(switcher[loc_in])
