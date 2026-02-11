# MCTS.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
from ChessNet import ChessNet
import ChessEngine

class MCTSNode:
    def __init__(self, prior, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.virtual_loss = 0

    def value(self):
        visit_count = self.visit_count + self.virtual_loss
        if visit_count == 0: return 0
        return self.value_sum / visit_count

    def select_child(self, c_puct=1.414):
        sqrt_total = np.sqrt(self.visit_count + self.virtual_loss + 1e-8)
        best_score = -float('inf')
        best_move, best_child = None, None
        
        for move, child in self.children.items():
            u = c_puct * child.prior * (sqrt_total / (1 + child.visit_count + child.virtual_loss))
            score = -child.value() + u
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

class MCTS:
    def __init__(self, model_state=None, device='cpu'):
        self.device = torch.device(device)
        self.model = ChessNet(input_channels=19).to(self.device)
        if model_state:
            self.model.load_state_dict(model_state)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda')
        self.memory = []

    def get_move_idx(self, move):
        start_idx = move.startRow * 8 + move.startCol
        end_idx = move.endRow * 8 + move.endCol
        base_idx = start_idx * 64 + end_idx
        if move.isPawnPromotion:
            promo_offset = {'Q': 0, 'R': 1, 'B': 2, 'N': 3}
            return (base_idx + promo_offset.get(move.promotionPiece, 0)) % 4672
        return base_idx % 4672

    def state_to_tensor(self, state):
        tensor = np.zeros((19, 8, 8), dtype=np.float32)
        piece_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
        for r in range(8):
            for c in range(8):
                piece = state.board[r][c]
                if piece != '--':
                    idx = piece_map[piece[1].lower()]
                    if piece[0] == 'w': tensor[idx, r, c] = 1.0
                    else: tensor[idx + 6, r, c] = 1.0
        if state.whiteToMove: tensor[12, :, :] = 1.0
        cr = state.currentCastlingRights
        if cr.wks: tensor[13, :, :] = 1.0
        if cr.wqs: tensor[14, :, :] = 1.0
        if cr.bks: tensor[15, :, :] = 1.0
        if cr.bqs: tensor[16, :, :] = 1.0
        tensor[17, :, :] = state.counter50Rule / 100.0
        if state.enpassantPossible:
            r, c = state.enpassantPossible
            tensor[18, r, c] = 1.0
        return torch.from_numpy(tensor).unsqueeze(0)

    def get_search_probs(self, state, num_simulations=400, training=True):
        root = MCTSNode(0)
        state_tensor = self.state_to_tensor(state).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

        valid_moves = state.getValidMoves()
        if not valid_moves: return np.zeros(4672), []

        if training:
            noise = np.random.dirichlet([0.3] * len(valid_moves))
            for i, move in enumerate(valid_moves):
                idx = self.get_move_idx(move)
                policy[idx] = 0.75 * policy[idx] + 0.25 * noise[i]

        for _ in range(num_simulations):
            node = root
            search_state = self.clone_game_state(state)
            while node.children:
                move, node = node.select_child()
                search_state.makeMove(move)

            v_moves = search_state.getValidMoves()
            if not v_moves or search_state.checkMate or search_state.staleMate:
                v = self.get_simple_value(search_state)
            else:
                s_tensor = self.state_to_tensor(search_state).to(self.device)
                with torch.no_grad():
                    p_logits, v_tensor = self.model(s_tensor)
                    p = F.softmax(p_logits, dim=1).cpu().numpy()[0]
                    v = v_tensor.item()
                for m in v_moves:
                    if m not in node.children:
                        m_idx = self.get_move_idx(m)
                        node.children[m] = MCTSNode(p[m_idx], parent=node)
            while node:
                node.visit_count += 1
                node.value_sum += v
                v = -v
                node = node.parent

        # 시각화를 위한 분석 데이터 추출
        search_info = []
        for move, child in root.children.items():
            search_info.append({
                'move': move,
                'count': child.visit_count,
                'value': -child.value()
            })
        search_info.sort(key=lambda x: x['count'], reverse=True)

        probs = np.zeros(4672)
        for move, child in root.children.items():
            probs[self.get_move_idx(move)] = child.visit_count
        sum_probs = np.sum(probs)
        return (probs / sum_probs if sum_probs > 0 else np.ones(4672) / 4672), search_info[:5]

    def clone_game_state(self, state):
        new_gs = ChessEngine.GameState()
        new_gs.board = [row[:] for row in state.board]
        new_gs.whiteToMove = state.whiteToMove
        new_gs.moveLog = state.moveLog[:]
        new_gs.whiteKingLocation = state.whiteKingLocation
        new_gs.blackKingLocation = state.blackKingLocation
        new_gs.enpassantPossible = state.enpassantPossible
        new_gs.counter50Rule = state.counter50Rule
        new_gs.checkMate = state.checkMate
        new_gs.staleMate = state.staleMate
        new_gs.currentCastlingRights = ChessEngine.CastleRights(
            state.currentCastlingRights.wks, state.currentCastlingRights.wqs,
            state.currentCastlingRights.bks, state.currentCastlingRights.bqs)
        return new_gs

    def get_simple_value(self, gs):
        if gs.checkMate: return -1.0
        return 0.0

    def get_best_move(self, state, training=True):
        sims = 400 if training else 200
        pi, _ = self.get_search_probs(state, num_simulations=sims, training=training)
        valid_moves = state.getValidMoves()
        if training:
            move_probs = [pi[self.get_move_idx(m)] for m in valid_moves]
            return random.choices(valid_moves, weights=move_probs)[0] if sum(move_probs) > 0 else random.choice(valid_moves)
        return max(valid_moves, key=lambda m: pi[self.get_move_idx(m)])