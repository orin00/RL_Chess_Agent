import chess.pgn
import torch
import os
import random
from tqdm import tqdm
import ChessEngine
from MCTS import MCTS

class PGNValidator:
    def __init__(self, device_name="cuda"):
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.mcts_handler = MCTS(None, self.device)

    # ChessEngine에서 Move 객체 기반 인덱스 계산한다.
    def get_move_idx(self, move):
        start_idx = move.startRow * 8 + move.startCol
        end_idx = move.endRow * 8 + move.endCol
        base_idx = start_idx * 64 + end_idx
        
        if move.isPawnPromotion:
            promo_offset = {'Q': 0, 'R': 1, 'B': 2, 'N': 3}
            piece = str(move.promotionPiece).upper() if move.promotionPiece else 'Q'
            return (base_idx + promo_offset.get(piece, 0)) % 4672
        return base_idx % 4672

    def validate_extraction(self, pgn_path, num_test_games=20):
        path = os.path.normpath(pgn_path)
        if not os.path.exists(path):
            print(f"Error: {path} 파일을 찾을 수 없습니다.")
            return

        all_games_data = [] 
        print(f"PGN 전체 스캔 시작 (제한 없음)")
        
        with open(path, mode='r', encoding='utf-8') as pgn:
            file_size = os.path.getsize(path)
            pbar = tqdm(total=file_size, desc="파일 스캔 중", unit="B", unit_scale=True)
            
            last_pos = 0
            while True:
                offset = pgn.tell()
                game = chess.pgn.read_game(pgn)
                if game is None: break
                
                # 진행 상황 업데이트
                current_pos = pgn.tell()
                pbar.update(current_pos - last_pos)
                last_pos = current_pos
                
                board = game.board()
                special_tags = {"PROMO": 0, "CASTLE": 0, "EP": 0}
                
                # 게임 내 특수 상황 카운트
                for move in game.mainline_moves():
                    if move.promotion:
                        special_tags["PROMO"] += 1
                    if board.is_castling(move):
                        special_tags["CASTLE"] += 1
                    if board.is_en_passant(move):
                        special_tags["EP"] += 1
                    board.push(move)

                all_games_data.append((offset, special_tags))
            pbar.close()

        if not all_games_data:
            print("\n파일 내에 분석할 수 있는 게임이 없습니다.")
            return

        print(f"\n스캔 완료: 전체 {len(all_games_data)}개 게임 발견.")
        print(f"그 중 {num_test_games}개를 무작위로 추출하여 상세 분석을 진행합니다.")

        # 샘플링된 게임 분석
        sampled_games = random.sample(all_games_data, min(len(all_games_data), num_test_games))
        
        with open(path, mode='r', encoding='utf-8') as pgn:
            for i, (offset, tags) in enumerate(sampled_games):
                active_tags = [f"{k}: {v}" for k, v in tags.items() if v > 0]
                tag_summary = ", ".join(active_tags) if active_tags else "Normal Game"
                
                print(f"\n--- {i+1}번째 분석 (Offset: {offset}) ---")
                print(f" 식별 결과 {tag_summary}")
                
                pgn.seek(offset)
                game = chess.pgn.read_game(pgn)
                gs = ChessEngine.GameState()
                node = game
                move_count = 0
                
                while not node.is_end():
                    next_node = node.variation(0)
                    move = next_node.move 
                    
                    promo = 'Q'
                    if move.promotion:
                        promo_map = {chess.QUEEN: 'Q', chess.ROOK: 'R', chess.BISHOP: 'B', chess.KNIGHT: 'N'}
                        promo = promo_map.get(move.promotion, 'Q')

                    start_sq = (8 - (move.from_square // 8 + 1), move.from_square % 8)
                    end_sq = (8 - (move.to_square // 8 + 1), move.to_square % 8)
                    
                    try:
                        engine_move = ChessEngine.Move(start_sq, end_sq, gs.board, promotionPiece=promo)
                        move_idx = self.get_move_idx(engine_move)
                        
                        valid_moves = gs.getValidMoves()
                        actual_move = next((m for m in valid_moves if m == engine_move), None)
                        
                        if actual_move:
                            if actual_move.isCastleMove or actual_move.isEnpassantMove or actual_move.isPawnPromotion:
                                final_type = "NORMAL"
                                if actual_move.isCastleMove: final_type = "CASTLE"
                                elif actual_move.isEnpassantMove: final_type = "EN PASSANT"
                                elif actual_move.isPawnPromotion: final_type = f"PROMO({actual_move.promotionPiece})"
                                
                                print(f"  > 수 {move_count+1:3d}: {move.uci():5s} | TYPE: {final_type:12s} | Index: {move_idx}")
                            
                            gs.makeMove(actual_move)
                            move_count += 1
                        else:
                            print(f"에러: {move.uci()} 가 유효하지 않음.")
                            break
                    except Exception as e:
                        print(f"중단: {e}")
                        break
                    node = next_node
                
                print(f"분석 완료 (총 {move_count}수)")

if __name__ == "__main__":
    validator = PGNValidator()
    pgn_file_path = r"lichess_db_standard_rated_2017-03.pgn\lichess_db_standard_rated_2017-03.pgn"
    validator.validate_extraction(pgn_file_path, num_test_games=100)