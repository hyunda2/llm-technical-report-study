# Switch Transformer — 필사 원문

사용자가 제공한 필사 원문 보존본임. 오탈자·괄호·추가 조사 표기를 유지했으며, 원문 대조와 교정은 [보고서](README.md)에 기록함.

- Switch Transformer&#x20;
  - deep learning에서 보통 모델은 “모든 입력”에 대해 “동일한 (모든) 파라미터”를 “재사용” 한다. Mixture of E
  - xpert 모델들은 이에 defy — 도전 각 입력 예시에 대해 “다른” 파라미터를 “선택”하여 사용한다.
  - 그 결과가 바로 sparsely-activated model이다. 장점은 outrageous한 파라미터와 동시에 constant한 compuataional cost.
  - 그러나, MoE의 문제는 바로 “complexity”, “communication cost — Expert가 서로 다른 GPU에 있음. 이 때 GPU 간 데이터 전송 비용”, “Training instability — 학습 불안정성. 특정 Expert만 선택되는 등 라우팅 불안정성으로 인한 문제. 학습이 비효율적으로 진행되거나 손실이 급증 혹은 수렴이 어려워지는 현상”
  - “We simplify the MoE routing algorithm”
  - and “Design intuitive improved models”
  - 처음으로 large sparse models들이 lower precision — bfloat 16 formats — 로 학습될 수 있다는 점을 보여준다.
  - T5-Base랑 T5-Large 기반 모델 디자인
  - multilingual setting에서도 유지
  - CC crawl corpus에 pre-training.
  - 21년도 논문임. 이 때 쯤에는 Large scale training이 flexible하고 powerful한 neural language models (심지어 LLM이라고 칭하지도 않음. 역사를 배워야 하는 이유다. 과거를 알면 흐름이 보인다) 의 효과적인 길이었음.
  - complicated algorithms 백날 만들어봐야 generous한 computational budget, data set size, parameter 라는 뒷배를 가진 simple architectures에 못미친다. (근데 이건 지금도 그럼 ㅋㅋㅋㅋ)
  - 문제는 computationally intensive하다는거.
  - “Inspired by the success of model scal, but seeking greater computational efficiency, we instead propose a —sparsely activate expert model—Switch Transformer”
  - “Sparsity”가 어디서 오냐? switch transformer의 case에서는, “neutral network weights”의 “subset”을 incoming example마다 activate 한다.
  - Figure 1 표가 매우 깔끔함. 깔끔한 idea, 깔끔한 figure, 위대한 논문??
  - 2021 당시에는 machine learning libraries, hardware accelerators는 “still cater” to dense matrix multiplications 였다.
  - 다시 말하지만 이 논문에서는 “MoE”를 “simplify”해서 “training stability”와 “computational benefits” yield하는 것을 목표로 했다.
  - MoE가 원래는 machine translation에서 성공했었다고 한다. (첨 알았다 ㄷㄷ)
  - 이 논문에서는 MoE가 broadly valuable in natural language 하다는걸 find한다고 한다.
  - Pre training. Fine tuning. Multi-task Training 세 학습에서 모두 효과적.
  - ..
  - 핵심은 FLOPs — floating point operations, 연산 수행 시 필요한 부동소수점 연산의 총 횟수 즉 계산량 — 를 늘리지 않고 파라미터 수만 늘리는 것
  - 기본적으로 FFN layer를 여러개의 독립적인 FFN으로 바꾼다. FFN 하나 = expert. 각 “토큰”은 이 중 하나를 통과한다.
  - Attention 연산은 그대로.
  - 가설은 “연산량과 무관하게 파라미터 수 자체가 성능에 중요하다”
  - Router: 토큰 표현이 들어오면 각 expert에 대한 점수를 계산. — token hidden state 랑 router weight matrix를 곱해 각 Expert의 Logit 계산. 이 때 router weight matrix의 크기는 expert 수 x 토큰의 hidden state 수
  - 이 때 gate 값 p_i가 곱해져서 router는 gradient 학습이 가능해진다. argmax 자체에는 gradient가 존재하지 않지만.
  - 기존 MoE(2017)은 router가 비교할 대상이 있어야 학습이 된다고 가정했다. —정확히는, top-1에서는 선택되지 않은 expert에 대한 학습 신호가 부족해 라우팅이 불안정해질 수 있다는 우려 때문에 여러 Expert를 선택하는 noisy top k gating을 제안.—. Switch trasnformer는 “top-1”으로도 충분히 학습된다는걸 보였다.
  - 이를 통한 이점은 세가지. router 연산이 준다. expert 당 처리할 토큰 수가 준다. gpu 간 communication이 단순해진다.
  - Switch transformer는 expert마다 받을 수 있는 토큰 수를 미리 정함. capasity — (배치 내 토큰 수 / expert 수)x capacity factor.— 이 때 cap factor가 1보다 크면 여유 버퍼가 생김. 특정 expert에 토큰이 몰려 capacity를 넘으면 초과 토큰은 드롭됨. 구체적으로는 해당 ffn 연산 건너뛰고 residual connection으로 바로 다음 층에 전달.
  - 이렇게 한 이유는 분산학습에서 텐서 크기가 정적이어야 하므로.
  - router를 그대로 두면 소수 expert에 토큰이 쏠리는 붕괴가 일어남. — 초기에 우연히 더 많은 토큰 배정받은 expert가 더 자주 학습되고, 라우터도 계속 해당 라우터를 선택하도록 학습되기 때문. — 이를 막기 위해 “보조 손실”을 더함.
  - 보조손실은 “expert로 실제 배정된 토큰 비율”에 “expert에 할당된 router 확률 평균”의 곱을 expert 전체에 대해 계산해서 더한다.
  - 배정이 완전 균등할 때 보조손실 값은 1. 이게 최소.
  - 별도의 학습 안정화 기법도 사용함. 1)selective precision. 모델 대부분을 bf16으로 학습하되 router 내부 연산, 특히 softmax는 f32로 계산.\
    &#x20;router 내부 연산은 지수 함수. 지수 함수는 저정밀도에서 불안정해지기 때문.\
    &#x20;2\)가중치 초기화 스케일을 기존의 1/10으로 줄임. 이는 초기 발산을 막음 — 이는 초기화 분포의 표준편차를 줄인다는 뜻, 이렇게 되면 초기 활성값과 라우터 logit이 과하게 커지는 것을 억제함. 학습 안정을 위함.— 3) 파인튜닝 시 파라미터 많은 expert의 내부에만 높은 dropout을 적용. 과적합을 줄임.
  - Switch transformer는 attention처럼 공유되는 dense param은 전 장치에 복사 (like data parral) expert만 장치별로 나눠 둠.
  - 그래도 있는 병목: all-to-all 통신. 모든 장치 동시에 통신하는 타이밍. 토큰마다 독립적으로 expert 선택하기 때문에, MoE 연산 한 번 할 때마다 router 계산 후 한번에 각 장치로 보낸다. “동시에.” expert 처리 후 에도 all-to-all로 한다. 이 때 expert capacity가 의미를 가진다.
  - 흥미로운 결과: sparse 모델을 dense 모델로 distill 하면 성능 향상분의 약 30%가 유지되었다고 함.
  - 남은 문제는 토큰 드롭, all-to-all 통신 비용, 파인튜닝 불안정성입니다. 이후 MoE 연구는 이 지점들을 보완하는 방향으로 발전했습니다. ST-MoE는 router z-loss로 안정성을 개선했고, Mixtral은 top-2로 돌아갔으며, DeepSeekMoE는 expert를 잘게 쪼개고 항상 활성화되는 shared expert를 두었습니다. DeepSeek-V3는 보조 손실 대신 router bias를 조정하는 auxiliary-loss-free 방식을 도입했습니다.
