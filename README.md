# Lotto Pattern Lab

한국 로또 6/45 당첨 번호를 검증하고 기본 패턴을 살펴보기 위한 Python 프로젝트입니다.

## 시작하기

Python 3.11 이상을 사용합니다. 저장소 루트에서 다음 명령을 실행합니다.

```powershell
python -m pip install -e .
lotto-pattern-lab 1 7 13 22 35 44
```

가상 환경은 저장소의 `.venv`를 사용할 수 있으며 Git에는 포함되지 않습니다.

## 기능

- 로또 6/45 당첨 번호와 보너스 번호 유효성 검사
- 홀수/짝수, 번호 합계, 구간별 번호 개수 요약
- 명령행에서 단일 회차 번호 분석

이 도구의 패턴 요약은 과거 번호를 설명하기 위한 통계이며 미래 당첨을 예측하거나 보장하지 않습니다.

## 개발

```powershell
python -m ruff check .
python -m pytest
```
