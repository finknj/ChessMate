import chess
import chess.engine
#import chess.uci
import os
import asyncio

CWD = '/home/pi/Desktop/ChessMate'
FEN = '8/2k5/r7/8/8/8/3PPP2/3K4 b'

async def main():
	
	transport, engine = await chess.engine.popen_uci('stockfish')
	board = chess.Board(FEN)
	

	while not board.is_game_over():
		result = await engine.play(board, chess.engine.Limit(time=2))
		board.push(result.move)
		print(result)
		break
		

	await engine.quit()



asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())