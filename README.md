## RL Chess Agent
강화학습을 기반으로 체스 게임을 수행하는 에이전트 프로젝트입니다.

### Project Structure
- **ChessEngine.py**: 체스 게임 규칙 및 로직 엔진
- **ChessNet.py**: Residual Network 기반 정책/가치 신경망 구조
- **MCTS.py**: Monte Carlo Tree Search 알고리즘 구현
- **SelfPlay.py**: 강화학습을 위한 자기 대국 로직
- **Pretrain.py**: Lichess PGN 데이터를 활용한 사전 학습 스크립트

### Model Weights
대용량 모델 파일(.pth)은 GitHub 용량 제한으로 인해 Hugging Face 저장소에서 배포됩니다.
- **Hugging Face ID**: `orin00/RL_Chess_Agent` 