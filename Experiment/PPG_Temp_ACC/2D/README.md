- `ResNet2_CNN1_CWT_SW10-1.py`
  - best val acc : 66.56%
  - 과적합 문제 심각
- `ResNet2_CNN1_CWT_SW10-1_2.py`
  - 가중치 고정
    - backbone layer freeze
  - batch 8 -> 16
  - dropout(0.3)
  - best vall acc : 75.47%

