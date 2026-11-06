---
week: 1
status: "진행 중"
author: "hyunda2"
study_scheduled: "2026-11-06T09:00:00+09:00"
study_started: ""
study_finished: ""
paper_title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
paper_url: "https://arxiv.org/abs/2101.03961"
source_verified_on: "2026-11-06"
---

# 01주차 — Switch Transformers 읽기 기록

> **핵심:** 토큰마다 Expert 하나를 선택해, 토큰당 계산량을 대체로 유지하면서 전체 파라미터 수를 늘리는 방법임.

| 항목 | 내용 |
| --- | --- |
| 제목 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity |
| 언제 | 최초 공개 2021-01-11 · arXiv v2 2022-04-30 · v3 2022-06-16 |
| Venue | 정식 출판은 학회가 아닌 **JMLR 23(120):1–39, 2022**임; arXiv는 preprint·개정본 제공 경로임 |
| 저자·소속 | William Fedus, Barret Zoph, Noam Shazeer · 논문 기재 소속 Google |
| 논문·코드 | [arXiv 이력](https://arxiv.org/abs/2101.03961) · [확인한 본문 v3](https://arxiv.org/html/2101.03961v3) · [JMLR](https://jmlr.org/papers/v23/21-0998.html) · [JAX/T5X](https://github.com/google-research/t5x) · [TensorFlow/Mesh](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py) |
| 진행 상태 | **진행 중** · 예정 일시 2026-11-06 오전 9시, 한국 시간 · 실제 시작일 미기록 · 완료일 미정 |

## 읽기 표시

| 표시 | 의미 |
| --- | --- |
| 📝 필사 정리 | 사용자 필사를 가독성 있게 재구성한 내용이며, 원문에 관한 서술은 해당 절·표와 대조함 |
| 💭 내 생각 | 계산·연도·약어를 제외한 괄호 속 생각과 사용자가 직접 남긴 감상임 |
| 🔎 추가 조사 / GPT의 해설 | 대시 사이 보충 메모와 GPT 설명임; 원문 사실·저자 주장과 구분함 |
| ✅ 원문 대조·교정 | 필사·GPT 답변에서 바로잡거나 조건을 보완한 내용임 |

[필사 원문](notes-original.md)과 [제공된 GPT Q&A 원문](qa-original.md)은 따로 보존함.
원본에는 교정 전 설명도 남아 있으므로, 정리·교정 결과는 이 글을 기준으로 읽음.

## 1. 공부 목적

- 이번 읽기 항목은 **Router의 Expert 선택 → Capacity Factor → Auxiliary Loss**의 연결을 이해하는 것임.
- 개인적인 학습 동기는 별도로 작성하지 않았으며 임의로 덧붙이지 않음.

## 2. 배경지식

### 📝 왜 모델을 크게 만들었나?

2021년 최초 공개된 논문임.

> 💭 **용어와 연구 흐름**
>
> 심지어 LLM이라고 칭하지도 않음. 역사를 배워야 하는 이유임. 과거를 알면 흐름이 보임.

🔎 이 감상은 이 논문의 “neural language models”라는 표현에 대한 메모임; 2021년 전체에 LLM이라는 용어가 없었다는 사실 주장으로 일반화하지 않음.

당시에도 계산 예산·데이터·파라미터 수를 충분히 늘린 단순한 구조가 강한 성능을 내는 흐름이 있었음.
GPT·T5처럼 Dense Transformer를 크게 만드는 방법은 효과적이지만 계산 비용도 큼.

> 💭 **스케일의 힘**
>
> 계산 예산·데이터·파라미터를 뒷받침한 단순한 구조가 복잡한 알고리즘을 이긴다는 이야기, 근데 이건 지금도 그럼 ㅋㅋㅋㅋ.

일반적인 Dense 모델은 입력마다 같은 연산 경로와 FFN 가중치를 사용함.
MoE, 즉 Mixture of Experts는 입력에 따라 일부 Expert만 선택함.
이렇게 일부 가중치만 활성화하는 모델이 **sparsely-activated model**임.

여기서 Expert는 독립적인 **FFN, 즉 토큰별 비선형 변환을 수행하는 신경망**임.
모든 파라미터가 매번 활성화되는 대신, 많은 파라미터를 보유하면서 일부만 계산에 사용한다는 점이 핵심임.

> 🔎 **추가 조사:** [DeepSeekMoE](https://arxiv.org/abs/2401.06066)는 Expert를 잘게 나누고 Shared Expert를 분리함; 이후 논문에서 Expert 구성의 변화를 비교할 수 있음.

근거: [§1 Introduction, §2](https://arxiv.org/html/2101.03961v3).

### 📝 Gap — MoE를 널리 쓰기 어려웠던 이유

| 문제 | 읽으면서 정리한 의미 |
| --- | --- |
| Complexity | 라우팅과 분산 실행 구조가 복잡함 |
| Communication cost | 다른 장치의 Expert로 토큰을 보내고 결과를 받는 비용이 발생함 |
| Training instability | 일부 Expert에 토큰이 몰리거나 라우팅·저정밀도 계산이 불안정해 학습 효율 저하, 손실 급증, 발산 등이 생길 수 있음 |

MoE는 언어 모델링·기계번역에서 이미 성과가 있었음.
이 논문은 번역을 넘어 사전학습·파인튜닝·멀티태스크 학습에서도 효율적인 스케일링을 보이려는 연구임.

> 💭 **MoE의 이전 성공**
>
> MoE가 기계번역에서 먼저 성과를 냈다는 점은 처음 알았음 ㄷㄷ.

> 🔎 **추가 조사 / GPT의 해설:** 당시 라이브러리와 GPU·TPU 같은 가속기는 밀집 행렬 곱에 맞춰져 있었으므로, 희소한 선택 자체뿐 아니라 이를 효율적으로 실행하는 구현도 중요했음.

근거: [§1, §6 Related Work](https://arxiv.org/html/2101.03961v3).

## 3. 핵심 아이디어

### 📝 연산량과 파라미터 수를 분리하기

저자의 가설은 **전체 계산량과 별개로 파라미터 수 자체가 중요한 스케일 축**이라는 것임.

- FLOPs는 부동소수점 연산의 총횟수, 즉 계산량임.
- Dense FFN을 여러 Expert로 바꾸고, 각 토큰은 그중 하나만 통과하게 함.
- 기본 구조의 Attention은 유지함.
- T5-Base·T5-Large 기반 구조를 사용하고 C4, 즉 Colossal Clean Crawled Corpus로 사전학습함.
- 다국어 실험에서도 효과를 확인함.

> 💭 **Figure 1**
>
> Figure 1 그래프가 매우 깔끔함. 깔끔한 아이디어, 깔끔한 Figure, 위대한 논문??

근거: [원문 Figure 1](https://arxiv.org/html/2101.03961v3).

> ✅ **범위 보완:** 계산량이 완전히 공짜로 고정되는 것은 아님; Expert를 늘리면 Router의 점수 계산, 메모리와 통신 부담은 남으며, 핵심은 Expert FFN의 토큰당 계산량을 대체로 유지하는 것임.

### 📝 Top-1으로 라우팅 단순화

기존 연구에는 Router가 둘 이상의 Expert를 비교해야 유의미한 학습 신호를 얻을 수 있다는 직관이 있었음.
Switch Transformer는 **Top-1만으로도 학습 가능함**을 실험으로 보여줌.

필사에서 정리한 이점은 다음 세 가지임.

1. 여러 Expert로 분배·결합하는 라우팅 계산이 줄어듦.
2. Top-2 대비 전체 Expert에 전달되는 토큰 사본 수와 필요한 Capacity를 줄일 수 있음.
3. 장치 간 통신과 라우팅 구현이 단순해짐.

모든 Expert의 Logit·확률은 여전히 계산함.
또한 각 Expert에 실제로 배정되는 토큰 수가 항상 균등하거나 정확히 절반이 된다는 뜻은 아님.

> 🔎 **추가 조사:** [Mixtral §2.1](https://arxiv.org/html/2401.04088)은 Top-2를 사용함; Switch의 결과가 모든 모델에서 Top-1만 정답이라는 뜻은 아님.

근거: [§2.1, Figure 2–3](https://arxiv.org/html/2101.03961v3).

### 한 토큰이 지나가는 경로

~~~mermaid
flowchart LR
    X["토큰 표현"] --> R["Router: Expert별 점수와 확률"]
    R --> I["Top-1 Expert 선택"]
    I --> C{"Capacity 이내?"}
    C -->|예| E["선택한 Expert 계산"]
    E --> G["Gate 확률을 출력에 곱함"]
    G --> O["잔차 경로와 결합해 다음 층으로"]
    C -->|초과| S["이 층의 Expert 계산 생략"]
    S --> O
~~~

🔎 FFN 경로의 개념도임; 토큰 드롭은 토큰을 문장에서 삭제한다는 뜻이 아니라 이 층의 Expert 계산을 생략한다는 뜻임.

## 4. 구조와 수식

### 4.1 Router — 무엇과 무엇을 곱하나?

📝 토큰 표현과 학습 가능한 Router 가중치를 곱해 Expert별 Logit, 즉 Softmax에 넣기 전 점수를 계산함.

원문의 열벡터 표기를 따르면 다음과 같음.

```math
x\in\mathbb{R}^{d},\quad W_r\in\mathbb{R}^{N\times d},\quad h(x)=W_rx
```

```math
p_i(x)=\frac{\exp(h_i(x))}{\sum_{j=1}^{N}\exp(h_j(x))},\qquad
i^*=\operatorname*{argmax}_i p_i(x),\qquad
y=p_{i^*}(x)E_{i^*}(x)
```

| 기호 | 의미 |
| --- | --- |
| `d` | 토큰 Hidden State의 차원 |
| `N` | Expert 수 |
| `p_i(x)` | 토큰 `x`를 Expert `i`로 보낼 Router 확률 |
| `E_i(x)` | Expert `i`가 계산한 출력 |

> 🔎 **추가 조사 / GPT의 해설:** Argmax를 직접 미분하는 것은 아니지만 선택된 Expert 출력에 곱한 Gate 확률을 통해 Router가 Gradient를 받음.

선택한 확률을 그대로 곱하는 것이 핵심임; Top-1 선택 후 Gate를 항상 1로 재정규화하면 이 Gate 경로의 Router 학습 신호가 사라짐.
이 설명은 위 식에 대한 GPT의 해설임.

> ✅ **표기 보완:** 필사의 `N×d`와 GPT 답변의 `d×N`은 벡터 방향이 다를 때 모두 가능한 표기임; 행벡터 구현은 `xW_r`를 사용하며, `d=768`, `N=8`이면 가중치가 $`768\times8`$임.

근거: [§2.1, 식 (1)–(2)](https://arxiv.org/html/2101.03961v3).

### 4.2 Capacity Factor — Expert가 받을 수 있는 토큰 수

📝 분산 구현에서는 Expert마다 처리할 최대 토큰 수를 미리 정함.

```math
C\approx\frac{T}{N}\times c
```

- `T`: 해당 라우팅 그룹의 토큰 수.
- `c`: Capacity Factor이며, 1보다 크면 평균 배정량보다 여유 공간을 둠.
- `C`: 정수로 정하는 Expert별 처리 한도이며 실제 정수화 방식은 구현을 확인해야 함.
- 한도를 넘은 토큰은 해당 Expert FFN 계산을 건너뛰고 Residual, 즉 잔차 경로로 전달됨.
- 여유 공간을 늘리면 드롭 가능성은 줄지만 Padding·메모리·통신 비용이 증가함.

> 🔎 **필사의 계산 예시 보완:** 토큰 100개, Expert 4개, `c=1.25`이면 기준값은 `31.25`임; 올림을 사용한다는 가정에서는 32자리이며, 모든 구현이 동일한 올림 규칙을 쓴다고 단정하지 않음.

> ✅ **범위 보완:** 정적 크기 요구는 이 논문의 TPU·Mesh TensorFlow/XLA 구현 맥락임; 모든 현대 분산 학습이 고정 Capacity를 요구하는 것은 아님.

> 🔎 **추가 조사:** [MegaBlocks](https://arxiv.org/abs/2211.15841)는 Block-sparse 실행으로 Token Drop 없이 가변적인 Expert 계산을 처리함.

근거: [§2.2, 식 (3), Figure 3](https://arxiv.org/html/2101.03961v3).

### 4.3 Auxiliary Loss — Expert 쏠림 완화

📝 Router를 그대로 두면 일부 Expert에 토큰이 집중되는 라우팅 붕괴가 생길 수 있어, 보조 손실을 더함.

```math
f_i=\frac{1}{T}\sum_{x\in\mathcal B}\mathbf{1}\!\left[\operatorname*{argmax}_j p_j(x)=i\right],
\qquad
P_i=\frac{1}{T}\sum_{x\in\mathcal B}p_i(x)
```

```math
L_{\mathrm{aux}}=\alpha N\sum_{i=1}^{N}f_iP_i
```

| 항목 | 의미 |
| --- | --- |
| `f_i` | Capacity 적용 전 Top-1 선택 기준으로 Expert `i`에 배정된 토큰 비율 |
| `P_i` | 배치에서 Expert `i`에 할당한 Router 확률의 평균 |
| `α` | 보조 손실의 세기이며 논문 기본값은 `0.01`임 |

> 🔎 **추가 조사 / GPT의 해설:** 초기에 선택을 많이 받은 Expert가 더 자주 학습되고 선택이 재강화되는 양의 피드백이 쏠림의 한 가지 설명임; 항상 붕괴한다는 뜻은 아님.

`f_i`는 Argmax로 정하므로 직접 미분하지 않으며, 미분 가능한 `P_i`를 통해 Router에 학습 신호를 줌.

> ✅ **필사 교정:** 균등하게 $`f_i=P_i=1/N`$이면 **`α`를 제외한 정규화 항은 1**, 실제 보조 손실은 **`α`**임; 두 통계가 다르므로 “1이 어떤 경우에도 엄밀한 전역 최솟값”이라고 단정하지 않음.

근거: [§2.2, 식 (4)–(6)](https://arxiv.org/html/2101.03961v3).
마지막 전역 최솟값에 대한 주의는 GPT의 수식 해석임.

> 🔎 **추가 조사·교정:** [DeepSeek-V3 §2.1.2](https://arxiv.org/html/2412.19437v2)는 Bias 조정으로 부하를 균형화하면서 작은 Sequence-wise 보조 손실도 함께 사용함; 필사의 “보조 손실 대신 Bias만 사용함”은 이 점을 보완해야 함.

### 4.4 학습 안정화

📝 저자는 대규모 Sparse 모델을 BF16으로 학습할 수 있음을 처음 보여줬다고 주장하며, 다음 기법을 함께 사용함.

| 기법 | 필사 정리 및 원문 대조 |
| --- | --- |
| Selective precision | 대부분은 BF16으로 계산하고 Router 내부의 로컬 계산은 FP32로 수행함; 통신에 쓰는 Dispatch·Combine 텐서는 BF16으로 되돌림 |
| 작은 초기화 스케일 | 초기 가중치 분포의 스케일을 줄여 학습 불안정을 완화함 |
| Expert dropout | 파인튜닝 시 Expert 내부 Dropout을 높여 과적합을 완화함; 모든 층의 Dropout을 일괄 높인다는 뜻은 아님 |

> 🔎 **추가 조사 / GPT의 해설:** Router 전체가 지수함수인 것은 아니며, Softmax의 지수·정규화 연산과 하드한 선택이 수치 오차에 민감하다는 점이 중요함.

초기화는 원문에서 다음과 같이 정의함.

```math
\sigma=\sqrt{\frac{s}{n}},\qquad
s:1\rightarrow0.1,\qquad
\frac{\sigma_{\mathrm{new}}}{\sigma_{\mathrm{old}}}=\frac{1}{\sqrt{10}}
```

> ✅ **필사·Q&A 교정:** 1/10로 줄이는 것은 스케일 `s`임; 표준편차 자체를 1/10로 줄인다는 설명과 0.02→0.002 예시는 이 논문의 정의에 맞지 않음.

작은 초기화가 활성값·Router Logit의 과도한 크기를 억제한다는 설명은 GPT의 직관적 해설임.
저자가 직접 확인한 결과는 초기화 스케일 축소가 안정성과 품질을 개선했다는 실험임.

> 🔎 **추가 조사:** [ST-MoE §3.3](https://arxiv.org/html/2202.08906v2)는 Router z-loss로 큰 Logit을 억제하고 학습 안정성을 다룸.

근거: [§2.4, Table 2–4](https://arxiv.org/html/2101.03961v3).

### 4.5 Expert 병렬화와 All-to-All

📝 Expert·Data 병렬화 구성에서는 공유되는 Dense 가중치를 복제하고 Expert 가중치를 장치별로 나눌 수 있음.
각 장치가 가진 토큰도 Router 선택에 따라 다른 장치의 Expert로 이동함.

> 💭 **분산 학습과의 연결**
>
> 공유 Dense 가중치를 복제하는 부분은 Data Parallel과 비슷하다고 이해함.

**토큰 묶음 전송 → Expert 계산 → 결과 수신** 과정에 All-to-All 통신이 필요함.
토큰마다 네트워크 요청을 한 번씩 보내는 방식이 아니라, 라우팅 결과를 묶어 장치 간 교환하는 방식임.

> ✅ **범위 보완:** 공유 Dense 가중치가 모든 구성에서 반드시 전 장치에 통째로 복제되는 것은 아님; 논문은 Model 병렬화까지 결합하는 구성도 다룸.

Capacity는 이 구현에서 통신·계산 버퍼의 크기와 메모리 사용량을 제어하는 역할을 함.
Top-1으로 통신 부담을 줄여도 All-to-All 병목 자체가 없어지지는 않음.

근거: [§5.4–5.5, Figure 9](https://arxiv.org/html/2101.03961v3).

## 5. 실험 결과

### Top-1과 Top-2 비교

아래는 원문 Table 1 일부임.
동일한 TPUv3 32코어를 사용하며, MoE·Switch는 128 Experts·격층 Expert 구성임.
두 Sparse 모델의 Capacity Factor는 1.0임.

| 모델 | 100k Step 품질: Negative log perplexity ↑ | 목표 품질까지 시간 ↓ | 처리량 ↑ |
| --- | ---: | ---: | ---: |
| T5-Base | -1.731 | 보고된 비교 구간에서 미달 | 1600 examples/s |
| MoE-Base, Top-2 | -1.572 | 80.1 h | 860 examples/s |
| Switch-Base, Top-1 | -1.561 | 62.8 h | 1000 examples/s |

목표 품질은 Negative log perplexity -1.50임.
높을수록 좋은 지표이므로 일반적인 Loss 감소와 방향이 반대임.
Switch는 T5-Base와 토큰당 FLOPs를 맞췄지만, 이 표의 Top-2 MoE까지 모두 같은 FLOPs인 것은 아님.

근거: [Table 1](https://arxiv.org/html/2101.03961v3).

### 📝 다국어·Distillation

- 다국어 효과는 mC4의 101개 언어와 mT5-Base 비교에서 확인함.
- Sparse 모델을 작은 Dense 모델로 Distill하면 **성능 향상분의 약 30%**를 유지할 수 있었다는 결과가 흥미로움.
- “원래 성능의 30%만 남았다”는 뜻은 아님.

Distillation, 즉 큰 모델의 출력을 작은 모델이 학습하게 하는 실험의 한 예는 다음과 같음.

| 모델 | 파라미터 | SuperGLUE 점수 ↑ |
| --- | ---: | ---: |
| T5-Base | 223M | 74.6 |
| Switch-Base Teacher | 7.4B | 81.3 |
| Distilled T5-Base | 223M | 76.6 |

기본 Dense 모델보다 오른 2.0점이 Teacher의 향상분 6.7점 중 약 30%에 해당함.
이는 원문 Table 8의 파인튜닝된 Teacher→Student 설정임.

근거: [§4.2–4.3, Table 8, Figure 7–8](https://arxiv.org/html/2101.03961v3).

### Conclusion — 어디까지 결론 낼 수 있나?

**[저자의 주장]** Top-1과 안정화 기법을 결합해, 전체 파라미터 수를 늘리면서도 효율적인 Sparse 학습이 가능함을 보여줌.

**[GPT의 해설·해석]** 핵심 구분은 **전체 모델 용량과 토큰당 계산량이 다르다**는 것임.
다만 모든 모델·하드웨어에서 같은 속도 향상을 얻거나, 파라미터를 늘리기만 하면 모든 작업이 개선된다는 결론은 아님.

### Limitations + Future work

- **[저자가 명시함]** Switch-XXL의 불안정성이 남아 있으며, 사전학습 품질 향상이 Downstream 성능으로 항상 그대로 이어지지는 않음.
- **[저자가 제안함]** 병렬화 설계·스케일링 관계, 서로 다른 크기의 Expert, Attention·다른 모달리티로의 확장 등이 후속 과제임.
- **[필사 기반 추가 해석]** 토큰 드롭·All-to-All 비용·파인튜닝 안정성을 이후 MoE를 읽을 때 이어서 살펴볼 항목으로 정리함.

근거: [§5.6, §8 Future Work](https://arxiv.org/html/2101.03961v3).

## 6. 이전 논문과 비교

### 이전 연구와의 연결

- Dense 계열의 GPT·T5·Transformer는 기본 경로의 가중치를 입력마다 공통으로 사용함.
- Shazeer et al.의 2017년 MoE는 Noisy Top-k로 여러 Expert를 선택하는 조건부 계산을 다룸.
- Switch는 그 흐름에서 Top-1의 학습 가능성과 효율을 검증한 연구로 읽을 수 있음.

근거: [Switch §1·2.1·6](https://arxiv.org/html/2101.03961v3), [2017 MoE](https://arxiv.org/html/1701.06538).

## 7. GPT 질의응답

제공된 Q1–Q30을 유지하고 답변은 한 문장 중심으로 압축함.
**A는 모두 GPT의 해설이며, 사용자 생각이나 저자의 직접 발언으로 취급하지 않음.**
아래에는 원문 대조에 따른 교정도 반영했으며, 교정 전 답변은 [Q&A 원문](qa-original.md)에 남김.

#### 기본 구조와 배경 — Q1–Q8

**Q1:** Communication Cost와 Training Instability란?

**A:** 통신 비용은 장치 간 토큰·결과 전송 부담이며, 학습 불안정성은 라우팅 쏠림·수치 오차 등으로 손실 급증이나 발산이 생기는 현상임.

**Q2:** Dense Transformer를 확장한 대표적인 연구는?

**A:** Transformer, GPT-1, T5, GPT-3가 주요 연결 연구이며, 각각 Self-Attention 구조·생성형 사전학습·Text-to-Text 통합·대규모 Few-shot 학습을 대표함.

**Q3:** Figure 1 왼쪽 그래프는 무엇을 의미함?

**A:** 토큰당 계산량을 대체로 유지하면서 Expert 수와 전체 파라미터를 늘릴 때 Test Loss가 낮아지는 경향을 보여줌.

**Q4:** Test Loss란 무엇이며 낮아진다는 것은?

**A:** 학습에 쓰지 않은 데이터의 예측 오차이며, 이 논문의 T5 기반 사전학습에서는 가려진 텍스트 구간을 더 잘 복원한다는 뜻임.

**Q5:** Machine Learning Libraries와 Hardware Accelerators란?

**A:** PyTorch·TensorFlow·cuBLAS 등은 연산을 구현·실행하는 소프트웨어이고, GPU·TPU는 이를 빠르게 실행하는 하드웨어임.

**Q6:** JAX는 어떤 언어이며 왜 JAX·TensorFlow 코드가 공개됐나?

**A:** JAX는 Python 기반 수치 연산·프로그램 변환 라이브러리이고, 논문은 JAX/T5X와 TensorFlow/Mesh 구현을 공개했으나 프레임워크 선택의 인과관계까지 원문으로 단정할 수는 없음.

**Q7:** FLOPs란?

**A:** 부동소수점 연산의 총횟수로, 모델 실행에 필요한 계산량을 나타내며 FLOP/s 같은 초당 처리 성능과는 구분함.

**Q8:** TPU와 GPU의 차이는?

**A:** GPU는 다양한 병렬 연산에 쓰이는 장치이고 TPU는 Google이 신경망의 대규모 행렬 연산에 특화해 설계한 가속기임.

근거·대조 범위: [Switch §1–2.3](https://arxiv.org/html/2101.03961v3), [JAX 공식 문서](https://docs.jax.dev/en/latest/).
Q4의 “다음 토큰 예측” 표현과 Q6의 프레임워크 선택 이유는 제공된 답변에서 보완함.

#### Routing — Q9–Q15

**Q9:** Router는 토큰 표현과 무엇을 곱함?

**A:** 학습 가능한 Router 가중치 행렬을 곱해 Expert별 Logit을 구하고 Softmax로 확률을 계산함.

**Q10:** Router 가중치 행렬의 형태는?

**A:** 행벡터 구현은 `d×N`, 원문의 열벡터 표기는 `N×d`이며 두 식은 전치 관계로 대응함.

**Q11:** 기존 MoE는 왜 Top-2 Routing을 사용했나?

**A:** 여러 Expert를 비교해야 Router가 유의미한 Gradient를 얻을 수 있다는 직관이 있었지만, 기존 Noisy Top-k가 반드시 `k=2`만을 뜻하는 것은 아님.

**Q12:** Noisy Top-k Gating은 왜 Noisy라고 부름?

**A:** Expert 점수에 Gaussian Noise를 더한 뒤 Top-k를 선택해 Expert 탐색과 부하 분산을 돕기 때문임.

**Q13:** Switch는 왜 Top-1으로도 충분하다고 판단했나?

**A:** Gate 확률을 통해 학습 신호를 받을 수 있으므로 단일 Expert 선택으로도 학습 가능하다는 가설을 세우고 효율·품질을 실험으로 검증했음.

**Q14:** Top-1의 효과는 어떻게 입증했나?

**A:** Table 1에서 동일한 하드웨어로 Top-1 Switch와 Top-2 MoE의 품질·처리량·목표 품질 도달 시간을 비교했음.

**Q15:** Top-1이면 왜 장치 간 통신 비용이 줄어듦?

**A:** 토큰을 최대 두 Expert로 보내고 받는 대신 하나의 Expert와 교환하므로 필요한 전송량과 분배·결합 부담을 줄일 수 있음.

근거: [Switch §2.1·Table 1](https://arxiv.org/html/2101.03961v3), [2017 MoE §2](https://arxiv.org/html/1701.06538).

#### Load Balancing — Q16–Q18

**Q16:** Router를 그대로 두면 왜 일부 Expert에 토큰이 몰리나?

**A:** 초기에 더 많이 선택된 Expert가 더 자주 학습되고 재선택되는 양의 피드백이 생길 수 있다는 해석임.

**Q17:** MoE 구조상 반드시 붕괴하는 문제인가?

**A:** 선택적 라우팅으로 발생하기 쉬운 문제지만 필연은 아니며 부하 균형화로 완화할 수 있음.

**Q18:** 보조 손실의 `f_i P_i` 항은 왜 균등 분배를 유도함?

**A:** 배정 비율 `f_i`를 직접 미분하지 않고 평균 확률 `P_i`를 조절해 많이 선택된 Expert의 상대적인 확률을 낮추는 학습 신호를 주기 때문임.

근거: [§2.2, 식 (4)–(6)](https://arxiv.org/html/2101.03961v3).
균등 상태의 손실값은 위의 `α` 포함 여부와 함께 읽어야 함.

#### 안정화 — Q19–Q22

**Q19:** 기존 MoE는 어떤 정밀도를 사용했고 왜 BF16이 어려웠나?

**A:** 이 논문이 비교한 GShard 계열은 안정성을 위해 FP32를 사용했으며, BF16에서는 Router의 Softmax와 선택이 수치 오차에 민감해질 수 있었음.

**Q20:** Selective Precision은 왜 필요한가?

**A:** Router 로컬 계산만 FP32로 높이고 나머지 계산·통신은 BF16을 유지해 안정성과 실행 효율을 함께 얻기 위함임.

**Q21:** 초기화 스케일을 줄이면 왜 안정될 수 있나?

**A:** 초기 활성값과 Logit이 과도하게 커질 가능성을 낮춘다는 직관으로 이해할 수 있으며, 논문은 작은 스케일의 안정성 개선을 실험으로 확인했음.

**Q22:** 초기 가중치 크기를 줄인다는 것은 구체적으로 무엇임?

**A:** 원문의 $`\sigma=\sqrt{s/n}`$에서 `s`를 1에서 0.1로 줄이는 것이므로, 표준편차는 기존의 `1/√10`로 줄어듦.

근거: [§2.4, Table 2–3](https://arxiv.org/html/2101.03961v3).
Q22는 제공된 답변의 “표준편차 1/10” 설명을 교정함.

#### 분산 학습과 Capacity — Q23–Q27

**Q23:** 왜 내 장치의 토큰이 다른 장치의 Expert로 이동함?

**A:** Expert 가중치가 여러 장치에 나뉘어 있고 Router는 토큰의 원래 장치가 아니라 Expert 점수를 기준으로 선택하기 때문임.

**Q24:** 토큰별로 독립적인 라우팅을 계산하기 때문인가?

**A:** 맞으며 같은 장치의 토큰도 서로 다른 Expert를 선택할 수 있으므로 목적 장치가 달라질 수 있음.

**Q25:** Expert Capacity는 왜 필요함?

**A:** 이 구현에서 Expert별 최대 처리 토큰 수를 정해 정적 텐서와 메모리·통신 버퍼를 확보하기 위함임.

**Q26:** All-to-All 통신과 Capacity는 어떻게 연결됨?

**A:** 토큰별 목적지는 달라도 고정된 Expert별 공간을 마련해 TPU/XLA 환경에서 크기가 정해진 텐서를 장치 간 교환할 수 있게 함.

**Q27:** 현대 MoE도 고정 Capacity가 반드시 필요한가?

**A:** 반드시 그렇지는 않으며 MegaBlocks처럼 Block-sparse 연산으로 Token Drop을 피하는 방식도 있지만, Expert 간 부하 불균형까지 없어지는 것은 아님.

근거: [Switch §2.2·5.4–5.5](https://arxiv.org/html/2101.03961v3), [MegaBlocks](https://arxiv.org/abs/2211.15841).
가변 크기 통신이 모든 현대 구현의 공통 조건이라는 식으로 일반화하지 않음.

#### 활용과 학습 방식 — Q28–Q30

**Q28:** MoE는 왜 기계번역에서 먼저 성공했나?

**A:** 대규모 다국어 번역은 언어별 패턴을 조건부 계산으로 다루면서 모델 용량을 늘리기 좋은 설정이었다는 해석임.

**Q29:** Pre-training·Fine-tuning·Multi-task Training에서 효율적인 Scaling을 보였다는 것은?

**A:** 사전학습·작업별 추가 학습·여러 작업의 공동 학습에서 Dense 비교군보다 유리한 품질·계산 효율의 확장 양상을 관찰했다는 뜻임.

**Q30:** Multi-task Training은 현대 LLM에서 사라졌나?

**A:** 여러 작업을 섞는 학습은 Instruction Tuning에도 이어져 데이터 혼합 비율과 작업 간 학습 간섭을 살펴야 하지만, 모든 Post-training을 같은 개념이나 단일한 발전 경로로 묶을 수는 없음.

근거·추가 해설: [Switch §1·4·6](https://arxiv.org/html/2101.03961v3), [FLAN 원문](https://arxiv.org/abs/2109.01652).
Q28의 성공 이유와 Q30의 연결은 GPT의 해석임.

### ✅ 필사·Q&A 교정 요약

| 원래 메모·답변 | 정리하면서 보완한 내용 |
| --- | --- |
| 균등하면 보조 손실의 최솟값은 1 | 정규화 항은 1, `α`를 곱한 실제 손실은 `α`이며 전역 최소에 대한 단정은 피함 |
| 초기화 분포의 표준편차를 1/10로 줄임 | 원문의 스케일 `s`가 1/10이며, 표준편차 비율은 `1/√10`임 |
| Test Loss 감소는 다음 토큰을 더 잘 예측한다는 뜻 | 이 논문은 T5 기반 가려진 텍스트 복원 목표를 사용함 |
| Router 행렬은 `N×d` 또는 `d×N` | 열벡터·행벡터 표기의 차이로 정리함 |
| 분산 학습에서는 정적 텐서가 필요함 | 해당 TPU/XLA 구현의 제약과 현대 MoE 일반을 구분함 |
| Dense 가중치는 항상 모든 장치에 복제됨 | Expert·Data 병렬화 구성의 설명이며 Model 병렬화 결합도 가능함 |
| DeepSeek-V3는 보조 손실 대신 Bias만 사용함 | 주된 Bias 균형화에 작은 Sequence-wise 보조 손실도 결합함 |

## 8. 참고자료

- 사용자 자료: [필사 원문](notes-original.md) · [GPT Q&A 원문](qa-original.md).
- Switch: [arXiv 이력](https://arxiv.org/abs/2101.03961) · [본문 v3](https://arxiv.org/html/2101.03961v3) · [JMLR 출판 정보](https://jmlr.org/papers/v23/21-0998.html).
- 관련 원문: [2017 MoE](https://arxiv.org/html/1701.06538) · [ST-MoE](https://arxiv.org/html/2202.08906v2) · [Mixtral](https://arxiv.org/html/2401.04088) · [DeepSeekMoE](https://arxiv.org/abs/2401.06066) · [DeepSeek-V3](https://arxiv.org/html/2412.19437v2) · [MegaBlocks](https://arxiv.org/abs/2211.15841) · [FLAN](https://arxiv.org/abs/2109.01652).
- 보충 개념: [JAX 공식 문서](https://docs.jax.dev/en/latest/).
- 원문 대조일: 2026-11-06이며 실제 학습 시작일과는 별도임.
