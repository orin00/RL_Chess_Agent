1. RL Chess Agent
  강화학습을 기반으로 체스 게임을 수행하는 에이전트 프로젝트입니다.  본 저장소에는 에이전트의 소스 코드와 학습 스크립트가 포함되어 있으며, 실제 모델 가중치 파일은 Hugging Face를 통해 배포됩니다.


2. Model Weights (Hugging Face)
  대용량 모델 파일(.pth)은 Hugging Face 저장소에서 다운로드하실 수 있습니다.  Hugging Face Repository: orin00/RL_Chess_Agent


3. Project Structure
  ChessEngine.py: 체스 게임의 규칙과 로직을 담당하는 엔진
  ChessMain.py: 게임 실행 및 사용자 인터페이스(UI) 메인 스크립트
  ChessNet.py: 에이전트의 핵심 신경망(Neural Network) 구조
  MCTS.py: Monte Carlo Tree Search 알고리즘 구현
  SelfPlay.py: 강화학습을 위한 자기 대국(Self-play) 로직
  Pretrain.py / pretrain_valid.py: 사전 학습 및 검증 스크립트
  Reward.py: 학습을 위한 보상 체계 설계
  images/: 체스 말 이미지 자산


4. Dataset Attribution & Credits
  본 프로젝트는 아래의 공개 데이터셋 및 데이터베이스를 활용하여 개발되었습니다.


4-1. Image Dataset
  Name: Chess Images (by Anmol Garg)
  License: Apache License 2.0
  Source: Kaggle Link

4-2. Pre-training Data (PGN)
  Source: Lichess Open Database
  Data: lichess_db_standard_rated_2017-03.pgn


5. License
  Source Code: 본 저장소의 코드는 LICENSE.txt를 따릅니다.
  Model Weights: Hugging Face에 업로드된 모델은 Apache License 2.0을 따릅니다.
