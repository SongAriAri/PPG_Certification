## Experiment Results

### SSM Encoder + Element-wise Sum Concat Fusion + Cross-Attention (baseline) 
- Accuracy : 25.72%
- (15 epoch)

### ResNet Encoder + Element-wise Sum Concat Fusion + Cross-Attention
- 방대한 양의 데이터 처리를 위해 ResNet의 Encoder로 교체
- Accuracy : 57.25%
- (30 epoch)

### ResNet Encoder + Concat Fusion + Cross-Attention
- embedding을 합치는 것이 아닌 옆으로 concat하는 방식
- Accuracy : 

