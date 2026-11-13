# 10주 학습 계획

- 주당 논문 1편, 총 10편을 공부합니다.
- 1주차 예정 일시(기록 보존): **2026년 11월 6일(금) 오전 9시(한국 시간, Asia/Seoul / UTC+09:00)**.
- 실제 학습 시작일: **미기록**. 2~10주차 일정은 미정이며 자동으로 정하지 않음.
- 진행표의 시작일·완료일은 실제 학습 기록입니다. 예정 일시가 정해져도 실제 시작일을 미리 채우지 않습니다.
- 학습 진행 상태: **1주차 완료(2026-11-06), 2주차 진행 중, 3~10주차 예정**.
- 서지 확인일: **2026-11-05**. 이 날짜는 자료 확인일이며 학습 시작일이 아닙니다.
- 요청한 주차 순서와 주제명을 유지합니다. 정식 논문명이 다르면 아래에 함께 표기합니다.
- 상태와 자료 확인은 별개임. 원문 링크 확인만으로 학습 진행 상태를 바꾸지 않음.

## 진행표

| 주차 | 요청한 학습 주제 | 집중해서 읽을 항목 | 학습 상태 | 시작일 | 완료일 | 글 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Switch Transformers | Router가 Expert를 선택하는 과정, Capacity Factor, Auxiliary Loss | 완료 | 미기록 | 2026-11-06 | [읽기 기록](posts/week-01/README.md) |
| 02 | Mixtral of Experts | Top-2 Routing, Weighted Sum, Total/Active Parameter | 진행 중 | 미기록 | 미정 | [읽기 기록](posts/week-02/README.md) |
| 03 | DeepSeekMoE | Expert를 작게 나누는 이유, Shared Expert가 필요한 이유, Expert Specialization의 효과 | 예정 | 미정 | 미정 | [초안](posts/week-03/README.md) |
| 04 | DeepSeek-V3 | GQA, MLA, KV Cache, MoE, Router, MTP | 예정 | 미정 | 미정 | [초안](posts/week-04/README.md) |
| 05 | Qwen3.8-Next | Hybrid Attention, Residual, Muon | 예정 | 미정 | 미정 | [초안](posts/week-05/README.md) |
| 06 | Kimi K3 | Attention Residuals, LatentMoE | 예정 | 미정 | 미정 | [초안](posts/week-06/README.md) |
| 07 | DeepSeek-V4.1-Flash | Long Context, KV Cache Compression | 예정 | 미정 | 미정 | [초안](posts/week-07/README.md) |
| 08 | DeepSeek-R1 | Reasoning RL, GRPO | 예정 | 미정 | 미정 | [초안](posts/week-08/README.md) |
| 09 | MiniMax-M2 Series | Agentic RL, Reward, Rollout | 예정 | 미정 | 미정 | [초안](posts/week-09/README.md) |
| 10 | Nemotron 3 Ultra | Training/Inference Efficiency | 예정 | 미정 | 미정 | [초안](posts/week-10/README.md) |

## 논문명과 원문 확인 기록

저자 제출 원문(arXiv) 또는 연구 기관이 공개한 기술 보고서를 기준으로 확인함. 확인되지 않는 이름·링크는 다른 모델로 대체하지 않고 **확인 필요**로 남김. 아래는 서지 정보·원문 위치의 확인 기록이며, 학습 진행 상태는 위 진행표와 해당 주차 글에 기록함.

### 01. Switch Transformers

- 정식 논문명: **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**
- 연구 기관/팀: Google
- 원문: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- 논문명·원문 확인: **확인됨** (2026-11-05)

### 02. Mixtral of Experts

- 정식 논문명: **Mixtral of Experts**
- 연구 기관/팀: Mistral AI
- 원문: [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- 논문명·원문 확인: **확인됨** (2026-11-05)

### 03. DeepSeekMoE

- 정식 논문명: **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**
- 연구 기관/팀: DeepSeek-AI
- 원문: [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- 논문명·원문 확인: **확인됨** (2026-11-05)

### 04. DeepSeek-V3

- 정식 논문명: **DeepSeek-V3 Technical Report**
- 연구 기관/팀: DeepSeek-AI
- 원문: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 논문명·원문 확인: **확인됨** (2026-11-05)
- 범위 메모: GQA를 포함한 학습 항목은 읽기 목표입니다. 배경·비교 개념과 실제 제안 구조를 혼동하지 않도록 원문에서 확인합니다.

### 05. Qwen3.8-Next

- 정식 논문명: **On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability**
- 연구 기관/팀: Qwen
- 원문: [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320)
- 논문명·원문 확인: **확인됨** (2026-11-05)
- 표기 메모: 요청명과 논문 제목은 Qwen3.8-Next이며, 논문 초록의 모델 표기는 Qwen3.8-Flash-Next입니다. 요청명을 유지하고, 학습할 때 모델 버전과 설명 범위를 구분합니다.

### 06. Kimi K3

- 정식 논문명: **Kimi K3: Open Frontier Intelligence**
- 연구 기관/팀: Moonshot AI
- 원문: [Kimi K3: Open Frontier Intelligence](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- 논문명·원문 확인: **확인됨** (2026-11-05)
- 추가 공식 근거: [공식 공개 안내](https://www.kimi.com/news/kimi-k3-open-source)

### 07. DeepSeek-V4.1-Flash

- 정식 논문명: **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression**
- 연구 기관/팀: DeepSeek-AI
- 원문: [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969)
- 논문명·원문 확인: **확인됨** (2026-11-05)
- 추가 공식 근거: [공식 발표](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)

### 08. DeepSeek-R1

- 정식 논문명: **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
- 연구 기관/팀: DeepSeek-AI
- 원문: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- 논문명·원문 확인: **확인됨** (2026-11-05)

### 09. MiniMax-M2 Series

- 정식 논문명: **The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence**
- 연구 기관/팀: MiniMax
- 원문: [The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence](https://arxiv.org/abs/2605.26494)
- 논문명·원문 확인: **확인됨** (2026-11-05)

### 10. Nemotron 3 Ultra

- 정식 논문명: **Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning**
- 연구 기관/팀: NVIDIA
- 원문: [Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2606.15007)
- 논문명·원문 확인: **확인됨** (2026-11-05)
- 추가 공식 근거: [NVIDIA 공식 보고서 PDF](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)

## 상태 갱신 규칙

- **예정** → **진행 중** → **리뷰 중** → **완료**를 기본 흐름으로 사용합니다.
- 실제로 공부를 시작했다는 사용자 확인이 있을 때 시작일과 상태를 기록합니다.
- 리뷰·토론과 작성자 확인이 끝났을 때만 완료로 바꾸고 실제 완료일을 기록합니다.
- `리뷰 중`은 글 내용을 검토하는 상태이며 PR 생성을 요구하지 않습니다.
- 파일 저장이나 커밋만으로 상태를 기계적으로 변경하지 않습니다. README와 해당 글의 상태를 함께 맞춥니다.
- 보류가 필요하면 이유를 기록하고 사용자와 다음 행동을 정합니다. 일정이나 대체 논문은 임의로 정하지 않습니다.
