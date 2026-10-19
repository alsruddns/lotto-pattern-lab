# Lotto Pattern Lab

로또 6/45 추첨 결과에 간헐적 규칙이나 통계적 상태 변화가 있는지 탐색하는 연구 도구입니다. 분석 결과는 가설 생성에만 사용하며 조작이나 외부 개입의 증거로 해석하지 않습니다. 공정한 무작위 추첨을 항상 대조 가설로 둡니다.

## 데이터

저장소의 `data/official/draws.csv`는 동행복권 결과 페이지에서 확인한 753회(2017-05-06)부터 1246회(2026-10-17)까지의 494개 결과 스냅샷입니다. CSV에는 1등 번호 6개와 날짜만 포함하며 보너스 번호는 제외했습니다. 옆의 `.manifest.json`에는 출처, 확인 시각, 회차 범위, 행 수, SHA-256이 기록되어 로딩할 때 파일 변경을 검사합니다. 최신 결과는 공식 결과 페이지에서 갱신할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m lotto_pattern_lab sync --start-round 753
```

로컬 `data/raw/lotto_draws_753_1246.csv`도 선택할 수 있습니다. 파일은 `draw_no,n1,n2,n3,n4,n5,n6` 형식이며 Git에서 제외됩니다. 로더는 `draw_no`와 `round`를 모두 지원합니다. 753~1246회 494개가 연속되고 각 회차에 유효한 6개 번호가 있으면 `locally_validated_unreconciled`로 표시합니다. 이는 파일 내부 구조 검증을 뜻하며 공식 원본과 행별 대조를 마쳤다는 뜻은 아닙니다.

직접 CSV를 가져올 때 필요한 열은 `round` 또는 `draw_no`와 `n1,n2,n3,n4,n5,n6`이며 `date`/`draw_date`와 `bonus`는 선택 열입니다. 별도 검증 manifest가 없는 다른 업로드 파일은 `csv_unverified`로 표시하고, URL만으로 공식성을 인증하지 않습니다. 합성 데이터는 실제 추첨 데이터와 별도 유형으로 기록합니다. 기본 연구 구간은 2017-05-06의 753회 이후입니다. 날짜/회차 필터링 후 선택된 행만 모델 입력에 전달하므로 선택 구간 앞의 데이터는 과거 lag로 들어가지 않습니다.

## 설치 및 실행

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run src/lotto_pattern_lab/app.py
```

Streamlit 화면에서 로컬 실데이터, 공식 스냅샷, CSV, 합성 데이터 중 하나를 선택하고 회차 범위와 GP 설정을 조정할 수 있습니다. 날짜 열이 없는 CSV에는 회차 범위 필터가 적용됩니다. `간헐적 패턴 탐색`은 게이트가 활성화하는 회차에서만 규칙을 적용합니다. 진행 중인 세대별 상태와 부분 결과는 `experiments/`에 원자적으로 저장됩니다.

CLI에서도 같은 간헐적 엔진을 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m lotto_pattern_lab intermittent --csv data/official/draws.csv --population 32 --generations 10 --workers 4 --controls 20 --seed 12345
.\.venv\Scripts\python.exe -m lotto_pattern_lab intermittent --csv data/raw/lotto_draws_753_1246.csv --population 8 --generations 1 --workers 1 --controls 1 --seed 753
.\.venv\Scripts\python.exe -m lotto_pattern_lab intermittent --synthetic 180 --population 16 --generations 2 --workers 1 --controls 1 --seed 12345
```

기존 전체 회차 GP는 `search` 하위 명령으로 유지됩니다. 워커는 1~4개이며 기본값은 4입니다.

## 간헐 규칙 탐색과 평가

- DEAP는 예측 시점에 공개된 회차 번호와 직전 최대 10회 결과의 순서 불변 요약량을 입력으로 하는 선택 게이트와 출력 수식을 진화시킵니다. `int(abs(gate)) % 3`에 따라 회차를 비활성(0) 또는 규칙 1/2(1/2)로 분류하므로 두 규칙이 번갈아 적용될 수 있고 활성 간격은 불규칙할 수 있습니다. 학습 후 게이트와 공식은 검증·테스트 시점에 고정됩니다.
- 별도 한 번호 출력과 6개 번호 집합 출력 모델을 실행합니다. 6개 예측은 순서 없는 집합으로 비교하고 중복 출력은 결정적으로 보정합니다. 정렬된 번호 위치를 추첨 순서로 해석하지 않습니다.
- 연산자는 안전 덧셈·뺄셈·곱셈·나눗셈·나머지, 절댓값, 최솟값·최댓값과 제곱·세제곱을 지원합니다. 수식 깊이, 노드 수, 최소 활성 비율, 활성 표본 수 페널티로 복잡하거나 드문 규칙을 억제합니다.
- 자료는 시간순으로 학습·검증·독립 테스트로 나뉩니다. 탐색 적합도는 학습에, 규칙 선택은 검증에 사용합니다. 테스트 점수는 검증으로 선택된 규칙에 한해 마지막에 계산합니다. 비선택 후보에는 테스트 점수와 테스트 이벤트를 제공하지 않습니다. 활성 조건의 입력에는 목표 회차 번호와 이전 결과만 사용하며, 활성화된 모든 회차를 평가합니다.
- 점수는 활성 회차에서 맞힌 번호 수를 6/45 무작위 기대치와 비교하고 전체 구간 길이로 정규화합니다. 한 번호와 6개 집합 모델 모두 번호 일치 수, 활성 비율, 규칙별 적중률, 활성 회차 및 회차 간격을 기록합니다.
- 같은 길이의 공정한 6/45 합성 데이터마다 동일한 GP 탐색과 검증 선택을 다시 수행하고, 선택된 최고 후보의 독립 테스트 점수를 비교합니다. 경험적 p 값에는 +1 보정을 적용하며 반복 횟수와 해상도를 표시합니다. 대조 반복이 99회 미만이면 결과를 `tentative`로 표시합니다. 적은 반복 횟수의 p 값은 증거가 약합니다.

## 가설적 상태 분석

대각 공분산 Gaussian HMM은 학습 구간에만 적합하고, 이후 상태 확률은 과거 관측만 이용한 순방향 필터링으로 계산합니다. 뒤 관측으로 과거 상태를 사후 확정해 예측 점수로 보고하지 않습니다. 변화점 탐지기는 이전 회차 요약량의 두 이동 창 평균을 온라인 비교합니다. 상태 A~E 라벨은 계산 편의를 위한 가설적 분류이며 실제 추첨기 내부 상태를 뜻하지 않습니다.

## 실험 산출물 및 한계

각 실험 JSON은 데이터 출처·범위, 시드와 탐색 설정, 공식·게이트, 학습/검증/테스트 지표, 활성 회차, 무작위 대조군, 상태 추정과 실행 시간 및 진행 기록을 보관합니다. `experiments/`는 Git에서 제외됩니다.

`시스템 설정 · 저장 용량 관리` 화면에서 실험·체크포인트·로그 등 관리 폴더의 사용량과 경고 한도를 확인할 수 있습니다. 큰 완료 실험은 원본 상세 데이터를 gzip 파일로 보존하고 JSON에 요약과 SHA-256 참조를 남기는 방식으로 압축할 수 있습니다. 실험 상세를 불러오면 압축 파일을 검증한 뒤 원래 구조로 복원합니다. 실행 중 실험은 압축하지 않으며 진행 기록 스냅샷에는 최근 25개 항목과 전체 개수를 보관합니다.

오래된 체크포인트 정리는 미리보기와 별도 확인 뒤 실행합니다. 최근 5개(설정으로 늘릴 수 있음), 실행 중이거나 재개 가능한 체크포인트, 완전 일치 발견 기록이 있는 체크포인트는 정리 대상에서 제외됩니다. 자동 정리는 기본 꺼져 있으며 켜도 검증된 종료 체크포인트만 정리합니다. 실험·추천 원본과 압축 상세 파일은 자동 삭제하지 않습니다. 저장 한도는 경고용이며 파일 기록을 차단하지 않습니다.

선택 게이트는 GP가 만든 단일 수식을 세 가지 분류값(비활성, 규칙 1, 규칙 2)으로 해석하는 제한된 조건 공간이며, 각 규칙별 별도 게이트나 인과 추론은 구현하지 않았습니다. HMM/변화점 결과는 탐색 요약이며, 작은 데이터 또는 적은 대조 반복으로 조작이나 규칙을 결론내릴 수 없습니다.

## 검사

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```


## 지속 연구·다중 패턴 확장

- calendar_round 프로파일은 공식 검증 추첨일에서 연·월·일·요일·연중 일수·주차·날짜 수 변환과 회차 자릿수 특징을 추가합니다. 추첨일이 없거나 검증되지 않은 실데이터에는 날짜 특징을 적용하지 않습니다. 다음 회차 예측에는 사용자가 예정 추첨일을 입력해야 합니다.
- DEAP의 수식 문법에는 32비트 범위로 제한한 XOR/AND/OR/시프트, 자릿수 합·역순·디지털 루트, 소수 판정, 약수 개수, 피보나치 인덱스 연산을 추가했습니다. 공식 실행은 허용된 연산자 파서만 사용합니다.
- 6개 모델 이벤트에는 실제 집합과 일치 여부, 회차별 0~6개 적중 분포, 학습·검증·독립 테스트 구간별 3개 이상/4개 이상/5개 이상/6개 완전 일치를 기록합니다.
- data/database/patterns.sqlite는 발견 후보, 정규화 지문, 상태 변경 이력, 회차별 검증을 저장합니다. 과거 탐색 중 완전 일치는 retrospective_search_fit으로 분류하고 사전 예측 근거로 사용하지 않습니다. 이전 JSON에 실제 당첨번호가 빠져 있으면 저장된 이벤트 회차와 선택 데이터만 대조해 복원하며 원본 JSON은 변경하지 않습니다.
- 공식 CSV를 갱신할 때 *.history.jsonl에 이전/신규 데이터 해시와 회차 범위를 기록합니다. CSV 및 manifest는 임시 파일에서 작성한 뒤 원자적으로 교체합니다.
- 지속 연구 대시보드 또는 CLI의 intermittent_gp 엔진으로 반복 탐색합니다. 현재 등록된 자동 탐색 엔진은 DEAP 간헐적 GP 한 종류입니다. 자기상관·FFT는 전체 시계열에 대한 탐색적 진단으로 제공되며, 예측 모델로 취급하지 않습니다.

Windows PowerShell 명령:

    .\.venv\Scripts\python.exe -m lotto_pattern_lab intermittent --csv data/official/draws.csv --feature-profile calendar_round --population 32 --generations 10 --workers 4 --controls 3

    .\.venv\Scripts\python.exe -m lotto_pattern_lab research --csv data/official/draws.csv --duration-minutes 30 --population 32 --generations 10 --workers 4 --controls 3 --checkpoint checkpoints/research.json

    .\.venv\Scripts\python.exe -m lotto_pattern_lab research --stop --checkpoint checkpoints/research.json

    .\.venv\Scripts\python.exe -m lotto_pattern_lab research --csv data/official/draws.csv --duration-minutes 30 --population 32 --generations 10 --workers 4 --controls 3 --checkpoint checkpoints/research.json --resume

체크포인트는 데이터 해시·조건·시드·진행 세대·실패 기록을 보존합니다. GP 개체군을 세대 중간에서 복구하지는 않으며, 재개할 때 다음 시드의 완전한 실험부터 시작합니다. 중단 요청은 현재 실험을 저장할 때까지 기다립니다. 긴 설정에서 한 실험 자체가 오래 걸리면 중단도 그만큼 지연됩니다.

PySR/Julia 백엔드, 완전한 재귀식·무제한 비정형 문법, 세대 중간 복원, 독립 Fourier 기반 규칙 진화는 아직 구현되지 않았습니다. 대조군 반복 수가 적으면 경험적 p값은 해상도가 낮고 잠정 결과로만 해석해야 합니다.


### 선택형 PySR 준비 상태

기존 .venv에 Julia 런타임을 자동 설치하지 않도록 PySR는 선택 의존성으로 분리했습니다. 별도로 설치하려면 다음 명령을 실행합니다.

    .\.venv\Scripts\python.exe -m pip install -e ".[symbolic]"

확인 시점에 PySR 저장소의 현재 프로젝트 메타데이터는 Python 3.10 이상을 선언하고 juliacall을 의존성으로 사용하므로 Python 3.13은 메타데이터 범위에 포함됩니다. 하지만 이 PC의 Julia 설치·초기화와 PySR 실제 탐색은 실행하지 않았으므로 Windows 11 런타임 호환성이 검증 완료된 상태는 아닙니다. 앱 시작 시 PySR를 import하지 않으며, 선택 백엔드 오류는 DEAP 기능과 분리해 표시합니다. [PySR 프로젝트 메타데이터](https://github.com/astroautomata/PySR/blob/master/pyproject.toml), [JuliaCall 의존성 안내](https://github.com/JuliaPy/PythonCall.jl/blob/main/docs/src/juliacall.md)
