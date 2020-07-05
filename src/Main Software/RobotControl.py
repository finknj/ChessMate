from MotorControl import MotorControl
from EndEffectorControl import EndEffectorControl
import Parameters as p
import time



class RobotControl:
    def __init__(self):
        self.var = 0
        self.finalPos = 0
        self.finalType = ""
        self.initialPos = 0
        self.initialType = ""
        self.capture = False

        self.MC = MotorControl()
        self.EE = EndEffectorControl()
        self.import_chessboard_dic(21)
        
    def import_chessboard_dic(self,chessboard_in):
        self.chessboardDictionary = chessboard_in

    # return the class of piece occupying the square 
    def get_square_type(self, indexKey):
        typeCode = self.chessboardDictionary.get(indexKey)
        switcher = {
            6: "empty",
            0:"bishop",
            1: "king",
            2: "knight",
            3: "pawn",
            4:"queen",
            5: "rooke",
            7:"bishop",
            8: "king",
            9: "knight",
            10: "pawn",
            11:"queen",
            12: "rooke"
            }
        return (switcher[typeCode])
##        self.chessboardDictionary.get(indexKey)
##        0 blue bishop
##        1 blue kning 
##        2 blue knight
##        3 blue pawn
##        4 blue queen
##        5 blue rooke 
##        6 nothing
##        7  yellow bishop 
##        8
##        9
##        10
##        11
##        12 yellow rooke

    # command to robot to mave a chess move from square arg1 to arg2 (a1,h8)
    def move_command(self, chessEngine_in):
        ts1 = time.time()
        
        self.initialPos = chessEngine_in[0]
        self.initialType = "knight" #self.get_square_type(self.initialPos)
        self.finalPos = chessEngine_in[1]
        self.finalType = "pawn"#self.get_square_type(self.finalPos)
 
        if (self.finalType == "empty") : self.capture = False
        else : self.capture = True

        # chek if a piece needs to be captured
        if self.capture == True:
            print ("Chess Capture: ",self.finalType,"(",self.finalPos,")")
            #get final pos coord
            goCoord = p.get_square_coord(self.finalPos)
            #move to final position
            self.MC.move_robot_arm(goCoord[0],goCoord[1])
            #pick up piece
            self.EE.set_elevation(self.finalType)
            #engage magnet
            self.EE.set_magnet(1)
            #pick up piece
            self.EE.set_elevation("top")
            #move to drop off
            self.MC.move_robot_arm(190,200)
            #drop piece
            self.EE.set_elevation(self.finalType)
            #disengage magnet
            self.EE.set_magnet(0)
            #raise ee
            self.EE.set_elevation("top")
            
        print("Chess Move: ",self.initialType," (",self.initialPos,") to (",self.finalPos,")")
        #get initial pos coord
        goCoord = p.get_square_coord(self.initialPos)
        #move to initial position
        self.MC.move_robot_arm(goCoord[0],goCoord[1]) 
        #pick up piece
        self.EE.set_elevation(self.initialType)
        #engage magnet
        self.EE.set_magnet(1)
        #pick up piece
        self.EE.set_elevation("top")
        #get final pos coord
        goCoord = p.get_square_coord(self.finalPos)
        #move to final position
        self.MC.move_robot_arm(goCoord[0],goCoord[1])
        #drop piece
        self.EE.set_elevation(self.initialType)
        #disengage magnet
        self.EE.set_magnet(0)
        #return
        self.EE.set_elevation("top")

        # return home
        self.MC.move_arm_home()

        ts2 = time.time()
        moveTime = float (ts2 - ts1)
        print("Chess move time: ",moveTime)


# example run 
RC = RobotControl()
#RC.import_chessboard_dic(chessboard_in)
RC.move_command(["a1","h8"])
