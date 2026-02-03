# SelfPlay.py
import ChessEngine
import MCTS as ChessAI
import Reward
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import warnings
import os
import time
import math

warnings.filterwarnings("ignore", category=FutureWarning)

def get_move_notation(move, reward=0.0):
    piece = move.pieceMoved[1].upper()
    start = f"{move.colsToFiles[move.startCol]}{move.rowsToRanks[move.startRow]}"
    end = f"{move.colsToFiles[move.endCol]}{move.rowsToRanks[move.endRow]}"
    notation = f"{piece}{start}->{end}"
    if move.pieceCaptured != '--':
        captured = move.pieceCaptured[1].upper()
        notation += f" x{captured}"
    if move.isPawnPromotion:
        notation += f"={move.promotionPiece}"
    if reward != 0:
        notation += f"({'+' if reward > 0 else ''}{reward:.2f})"
    return notation

def save_game_log(episode_idx, outcome, moves_list, total_time):
    if not os.path.exists("logs"): os.makedirs("logs")
    filename = f"logs/game_{episode_idx}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Outcome: {outcome}\n")
        f.write(f"Moves: {len(moves_list)} | Time: {total_time:.1f}s\n")
        f.write("-" * 50 + "\n")
        formatted_moves = []
        for i, move_str in enumerate(moves_list):
            turn = (i // 2) + 1
            prefix = "W" if i % 2 == 0 else "B"
            formatted_moves.append(f"{turn}.{prefix}({move_str})")
        for i in range(0, len(formatted_moves), 4):
            f.write("  ".join(formatted_moves[i:i+4]) + "\n")

def worker_process(worker_id, num_games, start_episode_idx, shared_model_dict, device_name, result_queue):
    device = torch.device(device_name)
    initial_weights = shared_model_dict['weights']
    mcts_handler = ChessAI.MCTS(initial_weights, device)
    reward_calc = Reward.RewardCalculator()

    for i in range(num_games):
        current_weights = shared_model_dict['weights']
        mcts_handler.model.load_state_dict(current_weights)

        gs = ChessEngine.GameState()
        game_data = [] 
        moves_list = [] 
        start_time = time.time()
        
        while True:
            valid_moves = gs.getValidMoves()
            is_draw = gs.staleMate or gs.counter50Rule >= 100 or gs.checkInsufficientMaterial()
            
            if not valid_moves or gs.checkMate or is_draw:
                if gs.checkMate:
                    outcome = "WHITE_WIN" if not gs.whiteToMove else "BLACK_WIN"
                    final_reward = reward_calc.get_final_reward(gs)
                else:
                    outcome = "DRAW"
                    final_reward = 0.0
                break

            if len(moves_list) >= 400:
                outcome = "DRAW (TIMEOUT)"
                final_reward = 0.0
                break

            state_tensor = mcts_handler.state_to_tensor(gs).cpu()
            # MCTS.py에서 제공하는 get_search_probs를 사용하여 데이터 수집
            move_probs = mcts_handler.get_search_probs(gs, num_simulations=200, training=True) 
            perspective = 1.0 if gs.whiteToMove else -1.0
            move = mcts_handler.get_best_move(gs, training=True)
            immediate_r = reward_calc.get_immediate_reward(gs, move)
            
            game_data.append([state_tensor, move_probs, perspective, immediate_r])
            moves_list.append(get_move_notation(move, immediate_r))
            gs.makeMove(move)

        processed_data = []
        for state, probs, perspective, imm_r in game_data:
            win_loss_v = perspective * final_reward
            v_combined = (win_loss_v * 0.7) + (imm_r * 0.3)
            v = math.tanh(v_combined)
            processed_data.append((state, probs, v))

        save_game_log(start_episode_idx + i, outcome, moves_list, time.time() - start_time)
        result_queue.put(("TRAIN_DATA", processed_data))
        result_queue.put(("DONE", worker_id, start_episode_idx + i))
        
        del game_data
        del processed_data

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    num_episodes, parallel_games = 5000, 8 
    games_per_worker = num_episodes // parallel_games
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    ai_master = ChessAI.MCTS(None, device_name)
    
    model_save_path = "chess_agent.pth"
    pretrained_path = "chess_model_pretrained.pth"
    
    if os.path.exists(model_save_path):
        print(f"강화학습 체크포인트를 로드합니다: {model_save_path}")
        ai_master.model.load_state_dict(torch.load(model_save_path, map_location=device_name))
    elif os.path.exists(pretrained_path):
        print(f"사전 지도학습 모델을 로드합니다: {pretrained_path}")
        ai_master.model.load_state_dict(torch.load(pretrained_path, map_location=device_name))
    else:
        print("새로운 모델로 학습을 시작합니다.")

    manager = mp.Manager()
    shared_model_dict = manager.dict()
    cpu_state = {k: v.cpu() for k, v in ai_master.model.state_dict().items()}
    shared_model_dict['weights'] = cpu_state
            
    result_queue = mp.Queue()
    processes = []
    main_pbar = tqdm(total=num_episodes, desc="자가 대국 진행 중")
    
    for i in range(parallel_games):
        p = mp.Process(target=worker_process, args=(i, games_per_worker, (i*games_per_worker)+1, 
                                                  shared_model_dict, device_name, result_queue))
        p.start()
        processes.append(p)
    
    completed = 0
    last_loss = 0.0 # 로스 기록용 변수 추가
    
    while completed < num_episodes:
        msg = result_queue.get()
        if msg[0] == "DONE":
            completed += 1
            # tqdm 설명창에 현재 메모리 크기와 마지막 로스값 표시
            main_pbar.set_description(f"학습 중 | Loss: {last_loss:.4f} | Memory: {len(ai_master.memory)}")
            main_pbar.update(1)
        elif msg[0] == "TRAIN_DATA":
            ai_master.memory.extend(msg[1])
            if len(ai_master.memory) >= 1024:
                # train_step은 보통 loss값을 리턴하므로 이를 활용
                current_loss = ai_master.train_step(batch_size=512)
                if current_loss is not None:
                    last_loss = current_loss
                
                cpu_state = {k: v.cpu() for k, v in ai_master.model.state_dict().items()}
                shared_model_dict['weights'] = cpu_state
                torch.save(ai_master.model.state_dict(), model_save_path)
        
        if len(ai_master.memory) > 100000:
            ai_master.memory = ai_master.memory[-70000:]
                
    for p in processes:
        p.join()
    main_pbar.close()
    print("전략적 자가 학습이 완료되었습니다.")