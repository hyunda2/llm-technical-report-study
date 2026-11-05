# LLM Technical Report Study

매주 논문 1편씩, 10주 동안 LLM의 구조와 학습 방법을 공부하고 한국어 Markdown 블로그 글로 기록합니다.

## 스터디 목적

- MoE, Attention, KV Cache, Reasoning 및 Agentic RL, 학습·추론 효율을 논문 원문으로 이해합니다.
- 논문의 주장, GPT의 해설, 작성자 자신의 생각을 구분하여 공부 기록을 남깁니다.
- Codex와 함께 브랜치 생성 → 커밋 → PR → 리뷰 → 명시적 승인 후 병합 흐름을 연습합니다.

## 참여자

- [hyunda2](https://github.com/hyunda2): 이 저장소의 소유자이자 스터디 참여자
- [yj4165](https://github.com/yj4165): 함께 공부하는 스터디 참여자

참여자 표기는 GitHub 협업자 초대나 저장소 접근 권한 부여를 뜻하지 않습니다.

참고 저장소: [yj4165/llm-technical-report-study](https://github.com/yj4165/llm-technical-report-study).
이 저장소는 hyunda2 계정에 새로 만든 독립 저장소입니다. 참고 저장소를 fork하거나 수정하지 않습니다.

## 진행 방식

- 전체 기간: 10주, 매주 1편. **1주차는 2026년 11월 6일(금) 오전 9시(한국 시간, Asia/Seoul) 예정**입니다.
- 실제 학습 시작일과 2~10주차 일정은 미정입니다. 예정 일시와 실제 시작일은 구분해 기록합니다.
- 초기 학습 상태는 10주 모두 **예정**입니다. 파일 생성이나 PR 병합만으로 학습 완료 처리하지 않습니다.
- [PLAN.md](PLAN.md)에서 주제·원문·진행 상태를 확인합니다.
- 매주 학습 기록용 Issue를 만들고 `study/week-01-switch-transformers`처럼 별도 브랜치에서 글을 작성합니다.
- [공통 템플릿](templates/study-post.md)과 해당 주차 초안을 사용합니다. 지금 있는 글은 학습 전 작성 틀입니다.
- 논문을 읽고 근거를 기록한 다음, 커밋하고 PR을 열어 내용·수식·인용·링크를 리뷰합니다.
- 리뷰 지적을 반영하고 **사용자가 특정 PR의 병합을 요청한 경우에만** 병합합니다. 자동 병합은 켜지 않습니다.
- 토론 결과와 학습 상태는 실제 진행에 맞춰 갱신합니다. 시작일·완료일·개인 의견을 임의로 채우지 않습니다.

## 글 목록

| 주차 | 학습 주제 | 글 | 상태 |
| --- | --- | --- | --- |
| 01 | Switch Transformers | [초안](posts/week-01/README.md) | 예정 |
| 02 | Mixtral of Experts | [초안](posts/week-02/README.md) | 예정 |
| 03 | DeepSeekMoE | [초안](posts/week-03/README.md) | 예정 |
| 04 | DeepSeek-V3 | [초안](posts/week-04/README.md) | 예정 |
| 05 | Qwen3.8-Next | [초안](posts/week-05/README.md) | 예정 |
| 06 | Kimi K3 | [초안](posts/week-06/README.md) | 예정 |
| 07 | DeepSeek-V4.1-Flash | [초안](posts/week-07/README.md) | 예정 |
| 08 | DeepSeek-R1 | [초안](posts/week-08/README.md) | 예정 |
| 09 | MiniMax-M2 Series | [초안](posts/week-09/README.md) | 예정 |
| 10 | Nemotron 3 Ultra | [초안](posts/week-10/README.md) | 예정 |

## 저장소 구성

```text
README.md
PLAN.md
AGENTS.md
templates/study-post.md
posts/week-01/README.md … posts/week-10/README.md
.github/PULL_REQUEST_TEMPLATE.md
.github/ISSUE_TEMPLATE/study-log.md
```

## Codex와 Git 작업

Codex는 작업 전에 [AGENTS.md](AGENTS.md)를 읽습니다. 이후 주차별 글과 운영 문서 수정은 브랜치와 PR로 관리합니다.
초기 저장소 구성만 `main`의 초기 커밋으로 반영했으며, PR 병합이나 협업자 초대는 수행하지 않았습니다.

요청 예시:

- “1주차 학습 Issue와 브랜치를 만들고, 내가 제공한 메모를 초안에 반영한 뒤 draft PR을 열어줘.”
- “이 PR의 원문 근거와 수식을 검토하고, 문제와 확인 필요 항목을 알려줘.”
- “리뷰를 반영한 PR #번호를 확인하고 병합해줘.”

GitHub 인증이나 권한 오류가 나면 실제 오류와 필요한 다음 행동을 보고합니다. 참조 저장소나 다른 로컬 프로젝트로 작업 대상을 바꾸지 않습니다.
