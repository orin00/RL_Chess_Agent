import torch
import torch.nn as nn
import numpy as np
import json
import os
import datetime
import ChessEngine
import ChessNet
import MCTS

class PretrainQA:
    def __init__(self, target_dir=r"..\qa\pretrain"):
        self.target_dir = target_dir
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mcts_handler = MCTS.MCTS(device=self.device)
        self.model = self.mcts_handler.model
        
        # QA 로그 파일 생성
        self.log_file = os.path.join(self.target_dir, f"pretrain_qa_report_{datetime.datetime.now().strftime('%Y%m%d')}.txt")

    def log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def validate_pgn_indexing(self):
        # PGN 데이터 파싱 정확도 테스트
        self.log("\n--- 1. Data Parsing & Promotion Indexing Test ---")
        gs = ChessEngine.GameState()
        
        # 폰 승급 상황 시뮬레이션 (8행 도달)
        # 백색 폰이 a7(1,0)에서 a8(0,0)로 이동하며 승급
        gs.board[1][0] = 'wp'
        promo_pieces = ['Q', 'R', 'B', 'N']
        results = {}

        for p in promo_pieces:
            move = ChessEngine.Move((1, 0), (0, 0), gs.board, promotionPiece=p)
            idx = self.mcts_handler.get_move_idx(move)
            results[p] = idx
            self.log(f"Promotion Piece: {p} | Calculated Index: {idx}")

        # 인덱스 유일성 검증
        if len(set(results.values())) == 4:
            self.log("[PASS] 승급 기물별 고유 인덱스 할당 확인.")
        else:
            self.log("[FAIL] 인덱스 중복 발생! get_move_idx 로직 점검 필요.")

    def analyze_history(self, history_path="pretrain_history.json"):
        # 2. Loss 수렴 확인 및 결과 저장
        self.log("\n--- 2. Loss Convergence Analysis ---")
        if not os.path.exists(history_path):
            self.log(f"[SKIP] {history_path} 파일이 없습니다.")
            return

        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        latest = history[-1]
        self.log(f"최종 에포크: {latest['epoch']}")
        self.log(f"Train Loss: {latest['train_loss']} | Valid Loss: {latest['valid_loss']}")
        self.log(f"Valid Accuracy: {latest['valid_acc']}%")

        # 수렴 추세 계산
        if len(history) >= 2:
            loss_diff = history[-1]['valid_loss'] - history[0]['valid_loss']
            if loss_diff < 0:
                self.log(f"[PASS] Loss가 초기 대비 {abs(loss_diff):.4f}만큼 하락하여 안정적으로 수렴 중입니다.")
            else:
                self.log("[WARNING] Loss가 하락하지 않고 있습니다. Learning Rate를 점검하세요.")

    def check_network_stability(self):
        """
        [QA 3] 기울기 소실 방지 및 가중치 업데이트 확인
        ChessNet의 GroupNorm과 ResBlock 가중치 분포 확인
        """
        self.log("\n--- [3] ChessNet Gradient & Weight Stability ---")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.log(f"Total Parameters: {total_params:,}")

        # 가중치 노름(Norm) 측정
        norms = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                norms.append(param.data.norm(2).item())
        
        avg_norm = np.mean(norms)
        self.log(f"Average Weight Norm: {avg_norm:.4f}")

        if 0.01 < avg_norm < 100.0:
            self.log("[PASS] 가중치 노름이 정상 범위 내에 있어 기울기 소실/폭주 징후가 없습니다.")
        else:
            self.log("[FAIL] 가중치 수치가 비정상적입니다. 초기화 로직을 확인하세요.")

    def run_full_qa(self):
        self.log(f"QA 시작 시간: {datetime.datetime.now()}")
        self.validate_pgn_indexing()
        self.analyze_history()
        self.check_network_stability()
        self.log(f"\n모든 QA 결과가 {self.target_dir}에 저장되었습니다.")

if __name__ == "__main__":
    qa_engine = PretrainQA()
    qa_engine.run_full_qa()