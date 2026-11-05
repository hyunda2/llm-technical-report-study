# LLM Technical Report Study

매주 논문 1편씩, 10주 동안 LLM의 구조와 학습 방법을 공부하고 한국어 Markdown 블로그 글로 기록합니다.

## 스터디 목적

- MoE, Attention, KV Cache, Reasoning 및 Agentic RL, 학습·추론 효율을 논문 원문으로 이해합니다.
- 논문의 주장, GPT의 해설, 작성자 자신의 생각을 구분하여 공부 기록을 남깁니다.
- Codex와 함께 글 작성 → 변경 내용 검토 → 작은 커밋 생성 → `main`에 반영하는 흐름으로 학습 기록을 관리합니다.

## 작성자

- [hyunda2](https://github.com/hyunda2): 개인 학습 기록 작성·관리

## 작성 원칙

- 음슴체 사용함.
- 핵심만 최소한으로 작성함. 이해에 꼭 필요한 내용은 판단해 추가함.
- 용어는 처음에 쉽게 설명하고, 핵심 방법의 작동 과정까지 설명함.
- 원문 확인 사실·저자의 주장·GPT의 해석을 구분함. 본문 접근 제한이나 수치 불일치는 명시함.
- 논문 링크·제목·파일을 보내면 원문 확인 후 한국어로 정리함. 일반 연구 논문은 [논문 정리 템플릿](templates/paper-summary.md) 순서 사용함.
- 실제 GPT 질의응답도 기록함. `Q:`·`A:` 각각 한 문장이 기본이며, 의미 보존에 꼭 필요한 경우만 문장 추가함.

## 진행 방식

- 전체 기간: 10주, 매주 1편. **1주차는 2026년 11월 6일(금) 오전 9시(한국 시간, Asia/Seoul) 예정**입니다.
- 실제 학습 시작일은 미기록이며 2~10주차 일정은 미정임. 예정 일시와 실제 시작일을 구분해 기록함.
- 현재 학습 상태는 **1주차 진행 중, 2~10주차 예정**임. 파일 생성이나 커밋만으로 학습 완료 처리하지 않음.
- [PLAN.md](PLAN.md)에서 주제·원문·진행 상태를 확인합니다.
- 주차별 글과 일정·운영 문서는 `main`에서 직접 수정합니다. 학습 기록용 Issue는 필요할 때 사용합니다.
- [공통 템플릿](templates/study-post.md)과 해당 주차 글을 사용함. 미학습 부분은 작성란으로 남겨 둠.
- 논문을 읽고 근거를 기록한 다음, Codex와 내용·수식·인용·링크 및 변경 내역을 검토합니다.
- 검토 지적을 반영하고 주차·작업별로 작은 커밋을 만들어 `main`에 반영합니다. **사용자의 별도 요청이 있을 때만 작업 브랜치와 PR을 만듭니다.**
- 토론 결과와 학습 상태는 실제 진행에 맞춰 갱신합니다. 시작일·완료일·개인 의견을 임의로 채우지 않습니다.

## 글 목록

| 주차 | 학습 주제 | 글 | 상태 |
| --- | --- | --- | --- |
| 01 | Switch Transformers | [읽기 기록](posts/week-01/README.md) | 진행 중 |
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
templates/paper-summary.md
posts/week-01/README.md … posts/week-10/README.md
.github/PULL_REQUEST_TEMPLATE.md
.github/ISSUE_TEMPLATE/study-log.md
```

## Codex와 Git 작업

Codex는 작업 전에 [AGENTS.md](AGENTS.md)를 읽습니다. 이 저장소는 `main` 하나를 기본으로 운영하며, 요청받은 변경을 검토한 뒤 직접 커밋하고 업로드합니다.
브랜치·PR·병합 연습은 별도 실습에서 다룹니다. 이 저장소에서 해당 작업을 하려면 사용자의 명시적 요청이 필요합니다. PR 템플릿은 요청 시 사용할 수 있도록 보관합니다.

요청 예시:

- “내가 제공한 1주차 메모를 초안에 반영하고, 검토한 뒤 `main`에 커밋해줘.”
- “이 글의 원문 근거와 수식을 검토하고, 문제와 확인 필요 항목을 알려줘.”
- “1주차 예정 일시를 계획표와 초안에 기록하고 `main`에 반영해줘.”

GitHub 인증이나 권한 오류가 나면 실제 오류와 필요한 다음 행동을 보고합니다. 다른 저장소나 로컬 프로젝트로 작업 대상을 바꾸지 않습니다.
