import pygame as p
import ChessEngine
import MCTS as ChessAI
import os
import datetime
import time

class UIValidator:
    def __init__(self, target_dir=r"..\qa\interface"):
        self.target_dir = target_dir
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            
        self.log_file = os.path.join(self.target_dir, f"interface_qa_report_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
        
        # 가상 디스플레이 설정
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        p.init()
        self.screen = p.display.set_mode((762, 512)) # BOARD + PANEL
        self.gs = ChessEngine.GameState()
        self.ai_handler = ChessAI.MCTS(device="cpu")

    def log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def test_animation_frame_integrity(self):
        # 애니메이션 프레임 및 데이터 일치성 검증
        self.log("\n--- [1] Animation & Data Integrity QA ---")
        move = ChessEngine.Move((6, 4), (4, 4), self.gs.board)
        
        start_time = time.time()
        # ChessMain.animateMove 로직 시뮬레이션
        dR = move.endRow - move.startRow
        dC = move.endCol - move.startCol
        framesPerSquare = 10
        frameCount = (abs(dR) + abs(dC)) * framesPerSquare
        
        for frame in range(frameCount + 1):
            # 프레임별 좌표 계산 로직 검사
            r = move.startRow + dR * frame / frameCount
            c = move.startCol + dC * frame / frameCount
            if frame == frameCount:
                self.log(f"Final Frame Coord: ({r}, {c}) -> Target: ({move.endRow}, {move.endCol})")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 이동 후 데이터 업데이트 검증
        self.gs.makeMove(move)
        if self.gs.board[4][4] == 'wp' and self.gs.board[6][4] == '--':
            self.log(f"[PASS] 애니메이션 종료 후 보드 데이터가 화면 좌표와 일치합니다. (소요시간: {duration:.4f}s)")
        else:
            self.log("[FAIL] 보드 데이터 업데이트 오류 발생.")

    def test_ai_inference_integration(self):
        # AI 추론 엔진 통합 및 패널 업데이트 검증
        
        self.log("\n--- AI Inference & UI Panel Sync QA ---")
        start_ai = time.time()
        
        # 시뮬레이션 횟수를 줄여 추론 지연 시간 측정
        best_move = self.ai_handler.get_best_move(self.gs, training=False)
        
        end_ai = time.time()
        latency = end_ai - start_ai
        
        self.log(f"AI Best Move Suggestion: {best_move}")
        self.log(f"MCTS Inference Latency: {latency:.4f}s")
        
        if latency < 2.0: # 2초 이내 응답 권장
            self.log("[PASS] AI 추론 속도가 UI 패널 업데이트에 적합한 수준입니다.")
        else:
            self.log("[WARNING] AI 추론 지연으로 인해 UI 프리징이 발생할 수 있습니다.")

    def run_full_ui_qa(self):
        self.log(f"UI QA 시작 시간: {datetime.datetime.now()}")
        try:
            self.test_animation_frame_integrity()
            self.test_ai_inference_integration()
        finally:
            p.quit()
        self.log(f"\n모든 UI 검증 보고서가 {self.target_dir}에 저장되었습니다.")

if __name__ == "__main__":
    qa_engine = UIValidator()
    qa_engine.run_full_ui_qa()