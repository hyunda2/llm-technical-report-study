# Switch Transformer 논문 읽기 — Q&A 정리

논문: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity\
링크: [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com)

## 1. MoE의 기본 구조와 등장 배경

Q1. Communication Cost와 Training Instability란?

- Communication Cost(통신 비용): Expert들이 서로 다른 GPU·TPU에 분산되어 있어, 토큰을 해당 Expert로 전송하고 결과를 다시 수집하는 과정에서 발생하는 장치 간 통신 비용.
- Training Instability(학습 불안정성): 특정 Expert에 토큰이 집중되거나 라우팅이 불안정해져 학습 효율이 떨어지고, 손실이 급증하거나 수렴이 어려워지는 현상.

Q2. 기존 Dense Transformer를 확장한 대표적인 연구는?

| 논문                    | 모델          | 핵심                            |
| --------------------- | ----------- | ----------------------------- |
| Radford et al. (2018) | GPT-1       | 생성형 사전학습                      |
| Raffel et al. (2019)  | T5          | Text-to-Text 통합 학습            |
| Brown et al. (2020)   | GPT-3       | 175B 규모 확장과 Few-shot Learning |
| Vaswani et al. (2017) | Transformer | Self-Attention 기반 아키텍처        |

이들은 MoE와 달리 모든 토큰에 동일한 파라미터를 사용하는 Dense Transformer 계열 연구다.

Q3. Figure 1 왼쪽 그래프는 무엇을 의미하는가?

Expert 수를 늘려 전체 파라미터 수를 증가시키면, 토큰당 계산량을 거의 일정하게 유지하면서도 Test Loss가 감소해 모델 성능이 향상된다는 의미다.

Q4. Test Loss란 무엇이며, 감소한다는 것은?

학습에 사용하지 않은 데이터에서 모델의 예측이 정답과 얼마나 다른지를 나타내는 값으로, 감소한다는 것은 처음 보는 텍스트의 다음 토큰을 더 정확하게 예측한다는 의미다.

Q5. Machine Learning Libraries와 Hardware Accelerators란?

- Libraries: PyTorch, TensorFlow, cuBLAS 등 머신러닝 연산을 구현하고 실행하는 소프트웨어.
- Hardware Accelerators: NVIDIA GPU, Google TPU 등 대규모 병렬 연산을 빠르게 수행하는 하드웨어.

당시 이들은 주로 Dense Matrix Multiplication(밀집 행렬 곱셈)에 최적화되어 있어 Sparse MoE를 효율적으로 실행하기 어려웠다.

Q6. JAX는 어떤 언어이며, 왜 JAX·TensorFlow 코드가 공개되었는가?

JAX는 프로그래밍 언어가 아니라 Python 기반 자동 미분·수치 연산 라이브러리이며, Switch Transformer 연구진이 Google TPU 중심의 연구 환경을 사용했기 때문에 TensorFlow·JAX 기반 구현을 공개했다.

Q7. FLOPs란?

FLOPs(Floating Point Operations)는 부동소수점 덧셈·곱셈 등의 총연산 횟수로, 모델 학습이나 추론에 필요한 계산량을 나타낸다.

Q8. TPU와 GPU의 차이는?

GPU는 그래픽 처리와 AI 연산 등에 활용되는 범용 병렬 연산 장치이고, TPU는 Google이 딥러닝의 대규모 행렬 연산에 특화해 설계한 AI 가속기다.

## 2. MoE Routing과 Switch Transformer의 핵심 아이디어

Q9. Router는 토큰 표현과 무엇을 곱하는가?

토큰의 Hidden State와 학습 가능한 Router Weight Matrix를 곱해 Expert별 Logit을 계산하고, Softmax를 적용해 선택 확률을 구한다.

Q10. Router Weight Matrix의 형태는?

\\[ W_r\in\mathbb{R}^{d\_{\text{model}}\times N\_{\text{experts}}} \\]

예를 들어 Hidden Dimension이 768이고 Expert가 8개라면, \\(768\times8\\) 행렬이며 토큰 벡터와 곱하면 Expert별 Logit 8개가 나온다.

Q11. 기존 MoE는 왜 Top-2 Routing을 사용했는가?

Top-1에서는 선택되지 않은 Expert에 대한 학습 신호가 부족하고 라우팅이 불안정해질 수 있다는 우려가 있었기 때문이지만, Shazeer et al.(2017)이 Top-2를 반드시 사용해야 한다고 주장한 것은 아니다.

Q12. Noisy Top-k Gating은 왜 Noisy라고 부르는가?

Expert별 라우팅 점수에 무작위 Gaussian Noise를 추가한 뒤 상위 k개 Expert를 선택하기 때문이며, 이는 다양한 Expert를 탐색하고 일부 Expert에 토큰이 집중되는 현상을 완화한다.

Q13. Switch Transformer는 왜 Top-1으로도 충분하다고 판단했는가?

Top-2가 반드시 필요하다는 이론적 근거가 부족한 반면 두 Expert를 사용하는 계산·통신 비용은 크기 때문에, Top-1만으로 충분한 성능을 얻을 수 있다는 가설을 세우고 실험으로 검증했다.

Q14. Top-1의 효과는 어떻게 입증했는가?

논문 Table 1에서 Top-1과 Top-2를 비교해, Top-1이 비슷하거나 더 좋은 언어 모델링 성능을 달성하면서 학습 속도는 더 빠르다는 결과를 제시했다.

Q15. Top-1을 사용하면 왜 GPU 간 Communication Cost가 감소하는가?

Top-2는 토큰을 최대 두 Expert로 전송하고 결과를 각각 받아야 하지만, Top-1은 하나의 Expert에만 전송하고 결과를 받으므로 장치 간 데이터 전송량이 감소한다.

## 3. Expert Load Balancing

Q16. Router를 그대로 두면 왜 일부 Expert에 토큰이 집중되는가?

초기에 우연히 많은 토큰을 배정받은 Expert가 더 자주 학습되고, Router도 해당 Expert를 계속 선택하도록 학습되는 양의 피드백이 발생할 수 있기 때문이다.

Q17. 이는 MoE 구조상 불가피한 문제인가?

토큰별 선택적 라우팅과 Expert 간 학습량 차이 때문에 구조적으로 발생하기 쉬운 문제지만, 반드시 붕괴하는 것은 아니며 Load Balancing Loss 등으로 완화할 수 있다.

Q18. Load Balancing Loss에서 \\(f_iP_i\\)를 최소화하면 왜 균등 분배되는가?

\\[ L\_{\text{aux}}=N\sum\_{i=1}^{N}f_iP_i \\]

- \\(f_i\\): Expert \\(i\\)에 실제 배정된 토큰의 비율.
- \\(P_i\\): Expert \\(i\\)에 대한 평균 Router 확률.

토큰이 많이 배정된 Expert는 \\(f_i\\)가 크므로, Loss를 최소화하면 해당 Expert의 \\(P_i\\)를 낮추는 방향으로 기울기가 발생하고 다른 Expert의 상대적 선택 확률은 높아져 균등 분배를 유도한다.

단, \\(f_i\\)는 Argmax 결과이므로 직접 미분하지 않고, 미분 가능한 \\(P_i\\)를 통해 Router를 학습한다.

## 4. 학습 안정성 개선

Q19. 기존 MoE는 어떤 Precision을 사용했으며, 왜 BF16이 어려웠는가?

기존 대규모 Sparse MoE 학습에서는 수치적 안정성을 위해 FP32를 사용하는 경우가 많았으며, BF16은 메모리와 연산 비용을 줄이지만 낮은 정밀도로 인해 Router의 확률 계산과 학습이 불안정해질 수 있었다.

Q20. Selective Precision은 왜 필요한가?

Router의 Softmax에서 작은 수치 오차도 Expert 선택을 바꿀 수 있으므로, 모델 대부분은 BF16으로 계산하되 Router 내부 연산은 FP32로 수행해 수치적 안정성을 높인다.

Q21. 가중치 초기화 스케일을 1/10로 줄이면 왜 학습이 안정되는가?

초기 가중치 크기를 줄이면 활성값과 Router Logit이 과도하게 커지는 것을 억제해, 초기 학습에서 특정 Expert로 확률이 집중되거나 값이 폭증하는 현상을 완화할 수 있다.

Q22. 초기 가중치 크기를 줄인다는 것은 구체적으로 무엇인가?

학습 시작 전 가중치를 무작위로 초기화할 때 분포의 표준편차를 기존의 1/10로 줄인다는 뜻으로, 예를 들어 표준편차 0.02를 0.002로 변경하는 것이다.

## 5. 분산 학습과 Expert Capacity

Q23. 왜 내 장치의 토큰이 다른 장치의 Expert로 라우팅되는가?

Expert들은 여러 GPU·TPU에 분산 저장되어 있고, Router는 토큰이 저장된 장치가 아니라 Expert의 적합도를 기준으로 선택하기 때문이다.

Q24. 이것이 토큰별로 라우팅을 계산하기 때문인가?

맞으며, MoE는 토큰마다 독립적으로 Expert를 선택하므로 같은 GPU에 저장된 토큰들도 서로 다른 장치의 Expert로 전송될 수 있다.

Q25. Expert Capacity는 왜 필요한가?

Expert Capacity는 각 Expert가 처리할 최대 토큰 수를 제한해, 장치 간 All-to-All 통신에 필요한 텐서 크기를 고정하고 메모리·통신 비용과 연산 지연을 예측 가능하게 만든다.

Q26. All-to-All 통신과 Expert Capacity는 어떻게 연결되는가?

All-to-All은 각 장치가 다른 장치들에 데이터를 교환하는 연산인데, 토큰별 라우팅 결과에 따라 전송량이 달라지므로 Switch Transformer의 TPU/XLA 환경에서는 고정된 Capacity를 사용해 정적인 텐서 크기를 확보했다.

예를 들어 Expert 4개에 토큰 100개를 배분하고 Capacity Factor를 1.25로 설정하면, Expert당 평균 25개에 여유분을 더해 약 32개를 처리하도록 공간을 확보한다.

- 남는 공간은 Padding으로 채운다.
- Capacity를 초과한 토큰은 Expert 계산을 건너뛰고 Residual 경로를 유지한다.
- 각 Expert의 최대 처리량이 제한되어 특정 장치의 과부하를 완화한다.

Q27. 현대 MoE에서도 Expert Capacity가 필수적인가?

반드시 그렇지는 않으며, MegaBlocks 같은 현대 GPU 기반 구현은 가변 크기 통신과 Block-sparse 연산으로 토큰을 드롭하지 않고 처리할 수 있지만, Expert 간 Load Imbalance 문제는 여전히 남는다.

## 6. MoE의 활용과 학습 패러다임

Q28. MoE는 왜 기계번역에서 먼저 성공했는가?

기계번역은 언어쌍·문법·어휘·문맥에 따라 필요한 처리 패턴이 다양하므로, Expert를 선택적으로 활용해 계산량을 억제하면서 모델 용량을 확장하는 MoE가 특히 대규모 다국어 번역에 유리했기 때문이다.

Q29. Pre-training, Fine-tuning, Multi-task Training에서 Superior Scaling을 보였다는 것은?

사전학습, 특정 작업에 맞춘 추가 학습, 여러 작업을 함께 학습하는 환경에서 모델 규모를 키웠을 때 기존 Dense Transformer보다 효율적으로 성능이 향상되었다는 의미다.

Q30. Multi-task Training은 현대 LLM에서 사라졌는가?

사라진 것이 아니라, 과거 번역·요약·분류 등 명시적인 여러 태스크를 함께 학습하던 방식이 현대에는 다양한 작업의 지시문과 정답을 혼합하는 Multi-task Instruction Tuning 및 범용 Post-training으로 발전했다.

과거에는 여러 태스크를 하나의 모델로 학습하는 것 자체가 중요한 연구 주제였다면, 현대에는 데이터 혼합 비율과 태스크 간 간섭(Negative Transfer)을 조절해 범용 능력을 향상시키는 것이 중요한 문제다.

## 핵심 정리

Switch Transformer의 핵심은 Top-1 Routing으로 토큰당 Expert 하나만 활성화해 계산·통신 비용을 줄이면서, Load Balancing Loss와 Selective Precision, 초기화 스케일 조정, Expert Capacity를 통해 대규모 Sparse MoE의 학습 안정성과 실행 효율을 확보했다는 것이다.

이 논문을 이해할 때 가장 중요한 구분은 \*\*전체 파라미터 수(Model Capacity)\*\*와 \*\*토큰당 실제 활성화되는 파라미터 수 및 계산량(Compute Cost)\*\*이 다르다는 점이다.