# Pretrain.py
import chess.pgn
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import json
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ChessNet import ChessNet
from MCTS import MCTS
import ChessEngine
from sklearn.model_selection import train_test_split

class ChessDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # state, move_idx, value 순서로 반환
        state, move, value = self.data_list[idx]
        return state.squeeze(0), move, torch.tensor([value], dtype=torch.float32)

class SLTrainer:
    def __init__(self, device_name="cuda"):
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.mcts_handler = MCTS(None, self.device)
        self.model = self.mcts_handler.model
        # 사전 학습에서는 정책과 가치를 동시에 최적화하기 위해 학습률을 조정할 수 있습니다.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.history_log = []

    def get_move_idx(self, move):
        """ChessEngine.Move 객체 기반 인덱스 계산"""
        start_idx = move.startRow * 8 + move.startCol
        end_idx = move.endRow * 8 + move.endCol
        base_idx = start_idx * 64 + end_idx
        
        if move.isPawnPromotion:
            promo_offset = {'Q': 0, 'R': 1, 'B': 2, 'N': 3}
            piece = str(move.promotionPiece).upper() if move.promotionPiece else 'Q'
            return (base_idx + promo_offset.get(piece, 0)) % 4672
        return base_idx % 4672

    def parse_result(self, result_str):
        """PGN 결과 문자열을 숫자 가치로 변환 (백 승: 1, 흑 승: -1, 무승부: 0)"""
        if result_str == "1-0":
            return 1.0
        elif result_str == "0-1":
            return -1.0
        else:
            return 0.0

    def collect_data_from_pgn(self, pgn_path, max_samples):
        """PGN에서 상태(State), 행동(Move), 결과(Value)를 함께 추출"""
        data_list = []
        path = os.path.normpath(pgn_path)
        if not os.path.exists(path):
            print(f"Error: {path} 파일을 찾을 수 없습니다.")
            return data_list

        offsets = []
        print(f"--- [1] PGN 파일 인덱싱 시작 ---")
        with open(path, mode='r', encoding='utf-8') as pgn:
            file_size = os.path.getsize(path)
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Indexing")
            while True:
                offset = pgn.tell()
                headers = chess.pgn.read_headers(pgn)
                if headers is None: break
                offsets.append(offset)
                pbar.update(pgn.tell() - offset)
            pbar.close()

        print(f"총 {len(offsets)}개의 게임 발견.")
        random.shuffle(offsets)
        
        print(f"--- [2] 데이터 추출 시작 (목표: {max_samples} Samples) ---")
        pbar_samples = tqdm(total=max_samples, desc="Collecting Samples")
        
        with open(path, mode='r', encoding='utf-8') as pgn:
            for offset in offsets:
                if len(data_list) >= max_samples:
                    break
                    
                pgn.seek(offset)
                game = chess.pgn.read_game(pgn)
                if game is None: continue
                
                # 가치 학습을 위한 게임 결과 추출
                game_result = self.parse_result(game.headers.get("Result", "1/2-1/2"))
                
                gs = ChessEngine.GameState()
                node = game
                
                while not node.is_end():
                    if len(data_list) >= max_samples:
                        break
                        
                    next_node = node.variation(0)
                    move = next_node.move
                    
                    promo = 'Q'
                    if move.promotion:
                        promo_map = {chess.QUEEN: 'Q', chess.ROOK: 'R', chess.BISHOP: 'B', chess.KNIGHT: 'N'}
                        promo = promo_map.get(move.promotion, 'Q')

                    start_sq = (8 - (move.from_square // 8 + 1), move.from_square % 8)
                    end_sq = (8 - (move.to_square // 8 + 1), move.to_square % 8)
                    
                    try:
                        state_tensor = self.mcts_handler.state_to_tensor(gs)
                        
                        engine_move = ChessEngine.Move(start_sq, end_sq, gs.board, promotionPiece=promo)
                        valid_moves = gs.getValidMoves()
                        actual_move = next((m for m in valid_moves if m == engine_move), None)
                        
                        if actual_move:
                            move_idx = self.get_move_idx(actual_move)
                            
                            # 현재 턴이 백이면 결과 그대로, 흑이면 반전시켜서 시점 일치 (Value Perspective)
                            perspective_value = game_result if gs.whiteToMove else -game_result
                            
                            data_list.append((state_tensor.cpu(), move_idx, perspective_value))
                            gs.makeMove(actual_move)
                            pbar_samples.update(1)
                        else:
                            break
                    except Exception:
                        break
                    node = next_node
                    
        pbar_samples.close()
        print(f"최종 수집된 샘플 수: {len(data_list)}")
        return data_list

    def train(self, pgn_path, epochs=2, batch_size=1024, max_samples=11000000):
        raw_data = self.collect_data_from_pgn(pgn_path, max_samples)
        if not raw_data: 
            print("데이터 수집에 실패했습니다.")
            return

        train_data, val_data = train_test_split(raw_data, test_size=0.01, random_state=42)
        train_loader = DataLoader(ChessDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ChessDataset(val_data), batch_size=batch_size)

        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss, epoch_p_loss, epoch_v_loss = 0, 0, 0
            epoch_acc = 0
            count = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for states, move_indices, values in pbar:
                states = states.to(self.device)
                move_indices = move_indices.to(self.device)
                values = values.to(self.device)

                self.optimizer.zero_grad()
                policy, value_pred = self.model(states)
                
                # 1. 정책 손실 (Cross Entropy)
                loss_p = F.cross_entropy(policy, move_indices)
                # 2. 가치 손실 (MSE)
                loss_v = F.mse_loss(value_pred, values)
                
                # 합산 손실 (가중치 조절 가능, 보통 정책 1 : 가치 1)
                total_loss = loss_p + loss_v
                
                total_loss.backward()
                self.optimizer.step()

                acc = (policy.argmax(dim=1) == move_indices).float().mean()
                epoch_loss += total_loss.item()
                epoch_p_loss += loss_p.item()
                epoch_v_loss += loss_v.item()
                epoch_acc += acc.item()
                count += 1
                pbar.set_postfix(L=f"{total_loss.item():.3f}", P=f"{loss_p.item():.3f}", V=f"{loss_v.item():.3f}", acc=f"{acc*100:.1f}%")

            v_loss, v_acc = self.evaluate(val_loader)
            print(f"\n--- Epoch {epoch+1} ---")
            print(f"Total Loss: {epoch_loss/count:.4f} (P:{epoch_p_loss/count:.3f}, V:{epoch_v_loss/count:.3f})")
            print(f"Valid Loss: {v_loss:.4f} | Valid Acc: {v_acc*100:.2f}%")
            
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(self.model.state_dict(), "best_pretrained_model.pth")
                print("최고 성능 모델 저장됨.")

            self.history_log.append({
                "epoch": epoch+1, 
                "train_loss": round(epoch_loss/count, 4),
                "valid_loss": round(v_loss, 4), 
                "valid_acc": round(v_acc*100, 2)
            })
            with open("pretrain_history.json", "w", encoding="utf-8") as f:
                json.dump(self.history_log, f, indent=4)

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss, val_acc, count = 0, 0, 0
        with torch.no_grad():
            for states, move_indices, values in val_loader:
                states = states.to(self.device)
                move_indices = move_indices.to(self.device)
                values = values.to(self.device)
                
                policy, value_pred = self.model(states)
                loss = F.cross_entropy(policy, move_indices) + F.mse_loss(value_pred, values)
                
                val_loss += loss.item()
                val_acc += (policy.argmax(dim=1) == move_indices).float().mean().item()
                count += 1
        return val_loss/count, val_acc/count

if __name__ == "__main__":
    trainer = SLTrainer(device_name="cuda")
    # 사용자의 실제 PGN 파일 경로로 수정 필요
    pgn_path = r"lichess_db_standard_rated_2017-03.pgn\lichess_db_standard_rated_2017-03.pgn"
    trainer.train(pgn_path, epochs=2, batch_size=1024, max_samples=11000000)