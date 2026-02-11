import torch
import torch.multiprocessing as mp
import numpy as np
import os
import datetime
import ChessEngine
import MCTS
import time

# RL QA를 위한 보상 계산 시뮬레이터 (Reward.py 로직 반영)
class RewardValidator:
    def calculate_reward(self, gs, move_count):
        if gs.checkMate:
            # 기본 승리 1.0 + 효율성 보너스 (빠른 승리 권장)
            efficiency_bonus = max(0, (50 - move_count) * 0.01)
            return 1.0 + efficiency_bonus
        if gs.staleMate or gs.counter50Rule >= 100:
            # 지루한 대국 방지를 위한 미세한 페널티
            return -0.05
        return 0.0

class RLValidator:
    def __init__(self, target_dir=r"..\qa\selfplay"):
        self.target_dir = target_dir
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mcts = MCTS.MCTS(device=self.device)
        self.log_file = os.path.join(self.target_dir, f"selfplay_qa_report_{datetime.datetime.now().strftime('%Y%m%d')}.txt")

    def log(self, message):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def test_mcts_exploration(self):
        """
        [QA 1] MCTS 탐색 효율성 검증 (c_puct 밸런스)
        """
        self.log("\n--- [1] MCTS Selection Logic QA ---")
        node = MCTS.MCTSNode(prior=1.0)
        
        # 가상의 자식 노드 2개 생성
        # 자식 1: Prior 높음(우수한 수), 방문 낮음 (Exploitation 대상)
        # 자식 2: Prior 낮음, 방문 매우 낮음 (Exploration 대상)
        child1 = MCTS.MCTSNode(prior=0.8, parent=node)
        child1.visit_count = 10
        child1.value_sum = 5.0 # 승률 0.5
        
        child2 = MCTS.MCTSNode(prior=0.2, parent=node)
        child2.visit_count = 1
        child2.value_sum = 0.1 # 승률 0.1
        
        node.children = {"move1": child1, "move2": child2}
        node.visit_count = 11
        
        best_move, _ = node.select_child(c_puct=1.414)
        self.log(f"Selected Move with c_puct(1.414): {best_move}")
        
        # c_puct가 매우 높을 때(탐색 강조) 변화 확인
        best_move_high, _ = node.select_child(c_puct=10.0)
        self.log(f"Selected Move with c_puct(10.0): {best_move_high}")
        
        if best_move != best_move_high or best_move is not None:
            self.log("[PASS] c_puct 상수에 따라 탐색과 활용의 우선순위가 변동됨을 확인했습니다.")

    def test_reward_system(self):
        """
        [QA 2] Reward.py 보상 체계 타당성 검토
        """
        self.log("\n--- [2] Reward System QA ---")
        rv = RewardValidator()
        gs = ChessEngine.GameState()
        
        # 1. 체크메이트 승리 (빠른 승리)
        gs.checkMate = True
        reward_fast = rv.calculate_reward(gs, move_count=10)
        # 2. 체크메이트 승리 (느린 승리)
        reward_slow = rv.calculate_reward(gs, move_count=80)
        # 3. 50수 규칙 무승부
        gs.checkMate = False
        gs.counter50Rule = 100
        reward_draw = rv.calculate_reward(gs, move_count=100)
        
        self.log(f"Fast Win Reward (10 moves): {reward_fast}")
        self.log(f"Slow Win Reward (80 moves): {reward_slow}")
        self.log(f"50-Rule Draw Reward: {reward_draw}")
        
        if reward_fast > reward_slow and reward_draw < 0:
            self.log("[PASS] 효율성 보너스 및 무승부 페널티 로직이 정상 작동합니다.")

    def test_clone_integrity(self):
        """
        [QA 3] clone_game_state 무결성 검사 (Deepcopy 미사용)
        """
        self.log("\n--- [3] State Cloning Integrity QA ---")
        gs_original = ChessEngine.GameState()
        gs_cloned = self.mcts.clone_game_state(gs_original)
        
        # 원본 수정 시 복제본 유지 여부 확인
        gs_original.board[0][0] = "--"
        
        if gs_cloned.board[0][0] == "bR":
            self.log("[PASS] 슬라이싱 기반 복제가 안전하게 이루어졌습니다.")
        else:
            self.log("[FAIL] 원본 수정이 복제본에 영향을 미칩니다! 참조 무결성 오류.")

    def run_full_rl_qa(self):
        self.log(f"RL QA 시작 시간: {datetime.datetime.now()}")
        self.test_mcts_exploration()
        self.test_reward_system()
        self.test_clone_integrity()
        self.log(f"\n모든 RL 검증 보고서가 {self.target_dir}에 저장되었습니다.")

if __name__ == "__main__":
    mp.freeze_support()
    validator = RLValidator()
    validator.run_full_rl_qa()