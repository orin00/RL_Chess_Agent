import pygame as p
import ChessEngine
import MCTS as ChessAI
import torch
import os

# 설정값
WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

def loadImages():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))

def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    p.display.set_caption("Chess AI - MCTS Player")
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    loadImages()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ai_handler = ChessAI.MCTS(None, device)
    model_path = "chess_agent.pth"
    if os.path.exists(model_path):
        try:
            ai_handler.model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            pass

    playerColor = None 
    waiting = True
    while waiting:
        screen.fill(p.Color("white"))
        drawText(screen, "Press 'W' for White, 'B' for Black")
        p.display.flip()
        for e in p.event.get():
            if e.type == p.QUIT:
                p.quit()
                return
            if e.type == p.KEYDOWN:
                if e.key == p.K_w:
                    playerColor = 'w'
                    waiting = False
                elif e.key == p.K_b:
                    playerColor = 'b'
                    waiting = False

    gs = ChessEngine.GameState()
    validMoves = gs.getValidMoves()
    moveMade = False
    animate = False
    running = True
    sqSelected = ()
    playerClicks = []
    gameOver = False
    
    while running:
        humanTurn = (gs.whiteToMove and playerColor == 'w') or (not gs.whiteToMove and playerColor == 'b')
        
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            
            elif e.type == p.MOUSEBUTTONDOWN:
                if not gameOver and humanTurn:
                    location = p.mouse.get_pos()
                    col, row = location[0] // SQ_SIZE, location[1] // SQ_SIZE
                    
                    if sqSelected == (row, col):
                        sqSelected, playerClicks = (), []
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)
                        
                    if len(playerClicks) == 2:
                        startSq = playerClicks[0]
                        endSq = playerClicks[1]
                        pieceSelected = gs.board[startSq[0]][startSq[1]]
                        
                        # --- 룩을 선택해서 캐슬링을 시도하는 경우 처리 ---
                        targetMove = None
                        if pieceSelected[1] == 'R': # 선택한 기물이 룩일 때
                            kingRow = 7 if pieceSelected[0] == 'w' else 0
                            kingCol = 4
                            # 룩을 킹 위로 옮기거나 킹 쪽으로 이동시켰을 때 캐슬링 수 탐색
                            if endSq == (kingRow, kingCol) or (endSq[0] == kingRow and abs(endSq[1] - startSq[1]) > 1):
                                # 엔진의 validMoves 중 해당 방향의 캐슬링 이동을 찾음
                                for m in validMoves:
                                    if m.isCastleMove:
                                        # 퀸사이드 캐슬링 (룩이 0번 열)
                                        if startSq[1] == 0 and m.endCol == 2:
                                            targetMove = m
                                            break
                                        # 킹사이드 캐슬링 (룩이 7번 열)
                                        elif startSq[1] == 7 and m.endCol == 6:
                                            targetMove = m
                                            break
                        
                        # 일반적인 이동(킹 선택 포함) 처리
                        # --- ChessMain.py 내 main 함수 내부의 이동 판정 로직 수정 ---
                        # 일반적인 이동(킹 선택 포함) 처리
                        if targetMove is None:
                            # 임시 이동 객체 생성 (사용자의 클릭 입력 기반)
                            tempMove = ChessEngine.Move(startSq, endSq, gs.board)
                            for i in range(len(validMoves)):
                                # 엔진이 생성한 유효한 수(validMoves) 중에서 
                                # 시작점과 끝점이 일치하는 실제 Move 객체를 찾습니다.
                                # 이렇게 해야 앙파상(isEnpassantMove) 속성이 올바르게 포함된 객체를 가져옵니다.
                                if tempMove.startRow == validMoves[i].startRow and \
                                   tempMove.startCol == validMoves[i].startCol and \
                                   tempMove.endRow == validMoves[i].endRow and \
                                   tempMove.endCol == validMoves[i].endCol:
                                    targetMove = validMoves[i]
                                    break
                        
                        # --- ChessMain.py 내 main 함수 내부의 프로모션 처리 로직 수정 ---
                        if targetMove:
                            if targetMove.isPawnPromotion:
                                # 사용자로부터 프로모션할 기물을 선택받음 (Q, R, B, N)
                                selectedPiece = getPromotionSelection(screen, clock, gs.whiteToMove)
                                
                                # validMoves 중에서 시작점, 끝점, 그리고 선택한 프로모션 기물이 일치하는 객체를 찾음
                                actualMove = None
                                for m in validMoves:
                                    if m.startRow == targetMove.startRow and m.startCol == targetMove.startCol and \
                                       m.endRow == targetMove.endRow and m.endCol == targetMove.endCol and \
                                       m.promotionPiece == selectedPiece:
                                        actualMove = m
                                        break
                                
                                # 일치하는 수를 찾은 경우에만 진행 (StopIteration 방지)
                                if actualMove:
                                    gs.makeMove(actualMove)
                                else:
                                    # 예외 상황 발생 시 기본 targetMove 실행
                                    gs.makeMove(targetMove)
                            else:
                                gs.makeMove(targetMove)
                            
                            moveMade = True
                            animate = True
                            sqSelected, playerClicks = (), []
                        else:
                            playerClicks = [sqSelected]

            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:
                    gs.undoMove()
                    gs.undoMove()
                    moveMade = True
                    animate = False
                    gameOver = False
                if e.key == p.K_r:
                    gs = ChessEngine.GameState()
                    validMoves = gs.getValidMoves()
                    sqSelected, playerClicks = (), []
                    moveMade = animate = gameOver = False

        if not gameOver and not humanTurn:
            ai_move = ai_handler.get_best_move(gs, training=False)
            gs.makeMove(ai_move)
            moveMade = True
            animate = True

        if moveMade:
            if animate and len(gs.moveLog) > 0:
                animateMove(gs.moveLog[-1], screen, gs.board, clock)
            validMoves = gs.getValidMoves()
            moveMade = animate = False

        drawGameState(screen, gs, validMoves, sqSelected)

        if gs.checkMate:
            gameOver = True
            drawText(screen, 'White wins' if not gs.whiteToMove else 'Black wins')
        elif gs.staleMate:
            gameOver = True
            drawText(screen, 'Stalemate')

        clock.tick(MAX_FPS)
        p.display.flip()

# highlightSquares 함수 수정 (룩 선택 시에도 캐슬링 칸 강조)
def highlightSquares(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color('blue'))
            screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))
            s.fill(p.Color('yellow'))
            
            pieceType = gs.board[r][c][1]
            for move in validMoves:
                # 1. 기물 자체의 일반 이동 강조
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol*SQ_SIZE, move.endRow*SQ_SIZE))
                # 2. 룩을 선택했을 때 캐슬링 가능한 킹의 위치 또는 이동 방향 강조
                if pieceType == 'R' and move.isCastleMove:
                    if c == 0 and move.endCol == 2: # 퀸사이드 캐슬링 가능
                        screen.blit(s, (4*SQ_SIZE, r*SQ_SIZE)) # 킹 위치 강조
                    elif c == 7 and move.endCol == 6: # 킹사이드 캐슬링 가능
                        screen.blit(s, (4*SQ_SIZE, r*SQ_SIZE))

def getPromotionSelection(screen, clock, isWhite):
    selection = None
    pieces = ['Q', 'R', 'B', 'N']
    color = 'w' if isWhite else 'b'
    box_width, box_height = SQ_SIZE * 4, SQ_SIZE + 20
    box_x, box_y = (WIDTH - box_width) // 2, (HEIGHT - box_height) // 2
    while selection is None:
        p.draw.rect(screen, p.Color("lightgray"), p.Rect(box_x, box_y, box_width, box_height))
        p.draw.rect(screen, p.Color("black"), p.Rect(box_x, box_y, box_width, box_height), 2)
        for i, p_type in enumerate(pieces):
            screen.blit(IMAGES[color + p_type], (box_x + i*SQ_SIZE, box_y + 10))
        p.display.flip()
        for e in p.event.get():
            if e.type == p.MOUSEBUTTONDOWN:
                location = p.mouse.get_pos()
                if box_y + 10 <= location[1] <= box_y + 10 + SQ_SIZE:
                    idx = (location[0] - box_x) // SQ_SIZE
                    if 0 <= idx < 4: selection = pieces[int(idx)]
        clock.tick(MAX_FPS)
    return selection

def animateMove(move, screen, board, clock):
    dR, dC = move.endRow - move.startRow, move.endCol - move.startCol
    frameCount = (max(abs(dR), abs(dC))) * 5
    if frameCount == 0: frameCount = 1
    rook_start_col, rook_end_col = -1, -1
    if move.isCastleMove:
        if move.endCol - move.startCol == 2: rook_start_col, rook_end_col = 7, 5
        else: rook_start_col, rook_end_col = 0, 3
    for frame in range(frameCount + 1):
        drawBoard(screen)
        for row in range(DIMENSION):
            for col in range(DIMENSION):
                piece = board[row][col]
                if piece != "--":
                    if not (row == move.endRow and col == move.endCol):
                        if not (move.isCastleMove and row == move.endRow and col == rook_end_col):
                            screen.blit(IMAGES[piece], p.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        r, c = (move.startRow + dR*frame/frameCount, move.startCol + dC*frame/frameCount)
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        if move.isCastleMove:
            rook_piece = 'wR' if move.pieceMoved[0] == 'w' else 'bR'
            c_rook = rook_start_col + (rook_end_col - rook_start_col)*frame/frameCount
            screen.blit(IMAGES[rook_piece], p.Rect(c_rook*SQ_SIZE, move.endRow*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)

def drawGameState(screen, gs, validMoves, sqSelected):
    drawBoard(screen)
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)

def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            p.draw.rect(screen, colors[((r + c) % 2)], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--": screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawText(screen, text):
    font = p.font.SysFont("Helvetica", 32, True, False)
    textObject = font.render(text, 0, p.Color('Gray'))
    textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)
    screen.blit(font.render(text, 0, p.Color("Black")), textLocation.move(2, 2))

if __name__ == "__main__":
    main()