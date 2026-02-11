import pygame as p
import ChessEngine
import MCTS as ChessAI
import torch
import os

# 설정값 확장 (우측 패널 추가)
BOARD_WIDTH = 512
PANEL_WIDTH = 250
WIDTH = BOARD_WIDTH + PANEL_WIDTH
HEIGHT = 512
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
    p.display.set_caption("Chess AI - MCTS Portfolio Demo")
    clock = p.time.Clock()
    loadImages()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ai_handler = ChessAI.MCTS(None, device)
    model_path = "chess_agent.pth"
    if os.path.exists(model_path):
        try:
            ai_handler.model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print("모델 로드 실패. 기본 가중치로 시작합니다.")

    gs = ChessEngine.GameState()
    validMoves = gs.getValidMoves()
    
    latest_search_info = []
    latest_value = 0.0
    moveMade = False
    animate = False
    running = True
    sqSelected = ()
    playerClicks = []
    gameOver = False
    
    # --- 진영 선택 로직 ---
    playerColor = None  # None이면 대기 상태
    while playerColor is None and running:
        drawStartScreen(screen)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            if e.type == p.KEYDOWN:
                if e.key == p.K_w:
                    playerColor = 'w'
                if e.key == p.K_b:
                    playerColor = 'b'
        p.display.flip()
        clock.tick(MAX_FPS)

    # --- 메인 대국 루프 ---
    while running:
        humanTurn = (gs.whiteToMove and playerColor == 'w') or (not gs.whiteToMove and playerColor == 'b')
        
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN and not gameOver and humanTurn:
                location = p.mouse.get_pos()
                if location[0] < BOARD_WIDTH:
                    col, row = location[0] // SQ_SIZE, location[1] // SQ_SIZE
                    if sqSelected == (row, col):
                        sqSelected, playerClicks = (), []
                    else:
                        sqSelected = (row, col)
                        playerClicks.append(sqSelected)
                    if len(playerClicks) == 2:
                        move = ChessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        for i in range(len(validMoves)):
                            if move == validMoves[i]:
                                gs.makeMove(validMoves[i])
                                moveMade = True
                                animate = True
                                sqSelected, playerClicks = (), []
                        if not moveMade:
                            playerClicks = [sqSelected]

        # AI Turn (시뮬레이션 횟수를 높여 더 정교하게 보이도록 함)
        if not gameOver and not humanTurn:
            # MCTS 시뮬레이션 및 데이터 추출
            pi, search_info = ai_handler.get_search_probs(gs, num_simulations=800, training=False)
            latest_search_info = search_info
            
            with torch.no_grad():
                _, v_tensor = ai_handler.model(ai_handler.state_to_tensor(gs).to(ai_handler.device))
                latest_value = v_tensor.item()

            ai_move = max(gs.getValidMoves(), key=lambda m: pi[ai_handler.get_move_idx(m)])
            gs.makeMove(ai_move)
            moveMade = True
            animate = True

        if moveMade:
            if animate and len(gs.moveLog) > 0:
                animateMove(gs.moveLog[-1], screen, gs.board, clock)
            validMoves = gs.getValidMoves()
            moveMade = animate = False

        # 화면 그리기
        drawGameState(screen, gs, validMoves, sqSelected)
        drawAnalysisPanel(screen, latest_search_info, latest_value)
        
        if gs.checkMate or gs.staleMate:
            gameOver = True
            drawText(screen, 'Checkmate!' if gs.checkMate else 'Stalemate')

        clock.tick(MAX_FPS)
        p.display.flip()

def drawStartScreen(screen):
    """시작 시 진영 선택 화면"""
    screen.fill(p.Color("#2c3e50"))
    font = p.font.SysFont("Helvetica", 30, True, False)
    text1 = font.render("Press 'W' to play as White", True, p.Color("white"))
    text2 = font.render("Press 'B' to play as Black", True, p.Color("white"))
    
    screen.blit(text1, (WIDTH // 2 - text1.get_width() // 2, HEIGHT // 2 - 40))
    screen.blit(text2, (WIDTH // 2 - text2.get_width() // 2, HEIGHT // 2 + 10))

def drawAnalysisPanel(screen, search_info, value):
    """우측 실시간 AI 분석 패널"""
    panel_rect = p.Rect(BOARD_WIDTH, 0, PANEL_WIDTH, HEIGHT)
    p.draw.rect(screen, p.Color("#1e1e1e"), panel_rect)
    
    font = p.font.SysFont("Consolas", 14)
    title_font = p.font.SysFont("Arial", 18, True)
    
    title = title_font.render("AI Thought Process", True, p.Color("yellow"))
    screen.blit(title, (BOARD_WIDTH + 15, 20))
    
    # 평가 바 (Value Gauge)
    p.draw.rect(screen, p.Color("#333333"), p.Rect(BOARD_WIDTH + 20, 60, 210, 15))
    fill_width = (value + 1) / 2 * 210
    color = p.Color("#27ae60") if value > 0 else p.Color("#c0392b")
    p.draw.rect(screen, color, p.Rect(BOARD_WIDTH + 20, 60, fill_width, 15))
    
    val_text = font.render(f"Value Score: {value:+.3f}", True, p.Color("white"))
    screen.blit(val_text, (BOARD_WIDTH + 20, 80))

    header = font.render(f"{'Move':<8}{'Visits':<8}{'Win%'}", True, p.Color("#bdc3c7"))
    screen.blit(header, (BOARD_WIDTH + 15, 120))
    
    y_pos = 150
    for info in search_info:
        # Move 객체의 좌표를 체스 기보(e2e4) 형태로 변환 (없으면 좌표로 표시)
        try:
            move_str = info['move'].getChessNotation()
        except:
            move_str = f"{info['move'].startRow}{info['move'].startCol}{info['move'].endRow}{info['move'].endCol}"
            
        visit_str = f"{info['count']}"
        win_rate = (info['value'] + 1) / 2 * 100
        
        text = font.render(f"{move_str:<8}{visit_str:<8}{win_rate:>5.1f}%", True, p.Color("white"))
        screen.blit(text, (BOARD_WIDTH + 15, y_pos))
        y_pos += 25

def drawGameState(screen, gs, validMoves, sqSelected):
    drawBoard(screen)
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)

def drawBoard(screen):
    colors = [p.Color("#eeeed2"), p.Color("#769656")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            p.draw.rect(screen, colors[((r + c) % 2)], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def highlightSquares(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color('blue'))
            screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))
            s.fill(p.Color('yellow'))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol*SQ_SIZE, move.endRow*SQ_SIZE))

def animateMove(move, screen, board, clock):
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 10
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR*frame/frameCount, move.startCol + dC*frame/frameCount)
        drawBoard(screen)
        drawPieces(screen, board)
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)

def drawText(screen, text):
    font = p.font.SysFont("Helvetica", 32, True, False)
    textObject = font.render(text, 0, p.Color('Black'))
    textLocation = p.Rect(0, 0, BOARD_WIDTH, HEIGHT).move(BOARD_WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)

if __name__ == "__main__":
    main()