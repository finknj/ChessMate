import chess
import chess.engine
#import chess.uci
import os
import asyncio

CWD = '/home/pi/Desktop/ChessMate'
FEN = '8/2k5/r7/8/8/8/3PPP2/3K4 b'


async def getEngineResults(board):
	transport, engine = await chess.engine.popen_uci('stockfish')
	result = await engine.play(board, chess.engine.Limit(time = 2))
	await engine.quit()
	return result.move



board = chess.Board(FEN)
result = asyncio.run(getEngineResults(board))


print('passed result: ' , result)
