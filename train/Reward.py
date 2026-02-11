import math

class RewardCalculator:
    def __init__(self):
        # 알파제로는 기물 가치를 학습을 통해 깨우치지만, 
        # 학습 가이드용으로 비숍(3.33)과 나이트(3.05)의 미세한 차이를 두는 것이 좋습니다.
        self.piece_values = {'p': 1.0, 'N': 3.05, 'B': 3.33, 'R': 5.63, 'Q': 9.5, 'K': 0.0}
        
        # 알파제로는 '중앙' 보다는 '활동성'을 중시합니다. 핵심 중앙 4칸에 집중합니다.
        self.center_squares = {(3, 3), (3, 4), (4, 3), (4, 4)}

    def evaluate_board(self, gs):
        """보드 상태를 승률 기대값으로 변환 보조"""
        if gs.checkMate:
            return -10.0 if gs.whiteToMove else 10.0
        if gs.staleMate:
            return 0.0

        score = 0.0
        # 보드 전체를 순회하되, 리스트 컴프리헨션이나 최적화된 루프를 고려할 수 있으나 
        # 가독성을 위해 기존 구조를 유지하며 미세 수정합니다.
        for r in range(8):
            for c in range(8):
                piece = gs.board[r][c]
                if piece != '--':
                    val = self.piece_values.get(piece[1], 0)
                    
                    if piece[1] == 'p':
                        dist = (7 - r) if piece[0] == 'w' else r
                        val += (dist * 0.01)
                    
                    if (r, c) in self.center_squares:
                        val += 0.05
                    
                    score += val if piece[0] == 'w' else -val
        return score

    def get_final_reward(self, gs):
        """최종 결과 보상: 더 빠른 승리에 가중치를 두어 신속한 결정을 유도"""
        if gs.checkMate:
            # 수의 길이에 따른 감쇠 함수를 강화하여 더 빨리 이길수록 보상을 크게 설정 (Efficiency 강화)
            # len(gs.moveLog)가 작을수록 보상이 1.0보다 커지도록 설계
            moves_made = len(gs.moveLog)
            efficiency_bonus = max(0.0, 0.5 * math.exp(-moves_made / 40.0)) 
            reward = 1.0 + efficiency_bonus
            return -reward if gs.whiteToMove else reward
        
        if gs.checkInsufficientMaterial() or gs.staleMate:
            return 0.0
            
        if hasattr(gs, 'counter50Rule') and gs.counter50Rule >= 100:
            return -0.05
            
        score = self.evaluate_board(gs)
        # 스쿼싱 계수를 유지하여 안정적인 기대값 산출
        return math.tanh(score * 0.1)

    def get_immediate_reward(self, gs, move):
        """즉각 보상: 결정 속도를 높이기 위해 무의미한 탐색에 대한 페널티를 강화"""
        reward = 0.0
        
        # 1. 기물 포획
        if move.pieceCaptured != '--':
            reward += self.piece_values.get(move.pieceCaptured[1], 0) * 0.05

        # 2. 앙파상 보너스
        if move.isEnpassantMove:
            reward += 0.05
            
        # 3. 특수 규칙
        if move.isPawnPromotion: 
            reward += 0.15 
        if move.isCastleMove: 
            reward += 0.1 
        
        # 4. 시간/수 페널티 (Time/Move Decay)
        # reward -= 0.001

        return reward