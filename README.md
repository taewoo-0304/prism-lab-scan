# Prism Lab — 3D 스캔

턴테이블 위 시료를 여러 각도에서 찍은 사진으로 **점군·메쉬·STL** 을 만든다.
젯슨이 촬영해서 보내고, 계산은 전부 PC(노트북)에서 한다.

```
젯슨 (jetson/)               PC (이 폴더)
  촬영 → 전송  ──────────▶   SAM 으로 물체 분리 → VGGT 3D 복원 → 정리 → 메쉬·STL
                              (선택) 가우시안 학습 → 렌더링 · 3D 뷰어
```

## 빠른 시작

**처음 한 번** — `VGGT-Server.bat` 더블클릭. 가상환경을 만들고 VGGT·SAM 을 설치한다.

**스캔** — `SCAN.bat` 더블클릭. 아무것도 묻지 않는다.

- 서버가 꺼져 있으면 알아서 켜고 기다린다
- 폴더를 안 주면 바탕화면에서 가장 최근 촬영 폴더를 잡는다 (사진 폴더를 드래그해도 된다)
- 결과는 `<사진폴더>\_scan\`

| 파일 | 내용 |
|---|---|
| `points.ply` | 색 있는 점군 |
| `mesh.ply` / `mesh.stl` | 메쉬 |
| `preview.png` | 4방향 미리보기 |
| `result.json` | 경로·점 개수·소요시간 |

133장 기준 약 17분, 34장(`--every 4`)이면 1~2분.

## 기능

### 1. 원클릭 스캔 — `SCAN.bat` / `scan_run.py`

사람이 입력할 게 없다. 시료는 항상 한 개고 턴테이블 위에 있으니 데이터에서 정한다.

- **SAM 프롬프트 자동 선택** — 후보를 12장 표본에 돌려 마스크를 가장 많이 잡는
  것을 고른다. 6개 촬영 폴더에서 실측:

  | 프롬프트 | 2058 | 2112 | 2119 | 2139 | 2044 | 2130 |
  |---|---|---|---|---|---|---|
  | `the object in the center` | 13 | 11 | 11 | 13 | 15 | 8 |
  | 3개 합집합 (현재 기본) | **16** | **16** | **16** | **14** | **16** | **16** |

- **점군 정리** (`object_clean.py`) — 밀도 군집으로 물체만 남기고, 물체에 붙은
  후광을 국소 밀도로 깎고, 잔잡음을 지운다.

  | | 점 | 대각선 | 중심거리 최대 |
  |---|---|---|---|
  | 서버 출력 | 1,807,052 | 7.68 | 4.06 |
  | 정리 후 | 840,497 | **0.465** | **0.22** |

  퍼센타일 필터로는 안 된다 — `dist_pct=92` 를 줘도 멀리 뭉친 배경이 최대 4.06
  으로 남았다. 군집은 거리가 아니라 연결성을 본다.

### 2. UI 연동 — `scan_run.py --json`

진행률을 JSON 한 줄씩 stdout 으로 낸다. `stage` 가 화면의 실행 단계 1·2·3 과 맞다.

```json
{"stage": 1, "state": "done",    "pct": 100, "images": 133, "device": "cuda"}
{"stage": 2, "state": "running", "pct": 70,  "msg": "물체만 남기는 중"}
{"stage": 2, "state": "done",    "ply": "...", "stl": "...", "n_triangles": 427662}
{"stage": 3, "state": "skipped", "msg": "분광기 미연결 — 재질 분석 건너뜀"}
```

`state` 는 `running` / `done` / `error` / `skipped`. 실패하면 종료 코드 1.

### 3. 정밀 복원 — `scan_run.py --precise`

VGGT 결과를 먼저 낸 뒤 COLMAP 덴스 MVS 까지 이어서 돌린다. **40~60분 더 걸린다.**
치수를 재거나 출력할 거면 이쪽 결과를 쓴다(아래 "한계" 참고).

### 4. 가우시안 렌더링 — `gof_prep.py` → `gof_pipe.py`

사진처럼 보이는 3D 렌더와 3D 뷰어를 만든다. **[Gaussian Opacity Fields](https://github.com/autonomousvision/gaussian-opacity-fields)
가 따로 필요하고, `gof_patch/` 의 패치를 적용해야 배경이 지워진다.** 설치는
`GOF-Setup.bat` → `GOF-Build.bat`, 자세한 내용은 [`gof_patch/README.md`](gof_patch/README.md).

```bash
python gof_prep.py <사진폴더> --pose colmap --sam --sam-prompt "the object in the center|the small object|the small machine part"
python gof_pipe.py <사진폴더>\_gof                         # 학습 + 메쉬
python %GOF_DIR%\render.py -m <사진폴더>\_gof\_model        # 렌더 (GOF 폴더에서)
```

⚠ **아직 명령 하나로 묶여 있지 않다.** UI 의 렌더링 버튼에 붙이려면 위 단계를
감싸는 진입점이 필요하다.

### 5. 3D 뷰어 HTML — `ply_to_splatsh.py` + `make_standalone_viewer.py`

학습된 가우시안을 **파일 하나짜리 HTML** 로 만든다. 더블클릭하면 브라우저에서
돌려볼 수 있고, 인터넷·서버·설치가 필요 없다. 외부 참조 0건.

```bash
python ply_to_splatsh.py <model>\point_cloud\iteration_15000\point_cloud.ply motor.splatsh
python make_standalone_viewer.py viewer\sh.html motor.splatsh 뷰어.html
```

구면조화 3차까지 넣어 금속 반사가 살아 있다. 바늘처럼 가는 가우시안(1.1%)은
걸러서 뺀다 — 안 빼면 물체 밖으로 검은 가시가 뻗는다.

⚠ `viewer/sh.html` 의 첫 시점(`HOME`)은 모터 촬영의 카메라 하나에서 뽑은 값이다.
다른 물체면 첫 화면 각도가 어색할 수 있다(마우스로 돌리면 된다).

### 6. 실루엣 카빙 (실험) — `visual_hull.py`

깊이 추정 없이 **마스크와 카메라 포즈만으로** 닫힌 메쉬를 만든다. VGGT 깊이
왜곡의 영향을 원천적으로 안 받는다. 모터 114장에서 6.5분, 결과가 항상 닫힌
입체라 3D 프린트에 바로 쓴다.

한계: 오목한 곳은 못 판다. 카메라가 안 닿은 방향(바닥 등)은 둥글게 남는다 —
모터 촬영에서 구면 방향의 31%가 가장 가까운 카메라와 40° 이상 떨어져 있었다.

⚠ 모듈만 있고 진입점이 없다. COLMAP 포즈(sparse)와 마스크를 넘겨 호출해야 한다.

## 한계 — 정직하게

**VGGT 메쉬는 굽은 껍데기로 나온다.** 점군을 다 정리하고 법선을 바깥으로 강제해도
같았다. 원인은 VGGT 깊이맵이다 — 한 프레임 안에서도 실제 형상과 TSDF 절단거리의
3.13배씩 어긋난다. 마스크를 133/133 완벽하게 잡아도 메쉬는 그대로였다. 근거는
[`experiments/README.md`](experiments/README.md).

**물체를 많이 탄다.** 같은 코드·설정으로 손선풍기(0831_2139)는 손잡이·헤드·날개까지
제대로 나왔고, 모터(0831_2058)와 마우스(0831_2112)는 안장처럼 휘었다.
어둡거나 금속 광택이 있는 작은 물체가 불리하다.

**형상이 필요하면** `--precise`(COLMAP, 40~60분) 또는 가우시안 경로를 쓴다.

**분광기(재질 분석, 실행 단계 1·3)는 구현돼 있지 않다.** `--json` 에서
`stage 3: skipped` 로 자리만 잡아뒀다.

## 폴더 구성

```
SCAN.bat               원클릭 진입점
scan_run.py            본체 (입력 없음, --json, --precise)
object_clean.py        점군 정리
server.py              VGGT 서버 (Flask, 포트 5000)
sam_mask.py            SAM 3 마스킹
colmap_pipe.py         COLMAP 정밀 경로
visual_hull.py         실루엣 카빙 (실험)
gof_*.py               가우시안 데이터셋·학습·설치
ply_to_splat*.py       가우시안 → 웹 뷰어 형식
make_standalone_viewer.py  뷰어 HTML 한 파일로
localtest.py           미리보기 이미지 (scan_run 이 쓴다)
make_drawing.py        점군 → STL, 도면 PDF (scan_run 이 쓴다)
rig_poses.py           리그 정기구학 → 카메라 mm 좌표 (자 측정 1회 필요)
launcher.py, setup_laptop.py, VGGT-*.bat   설치·실행

viewer/                3D 뷰어 HTML 템플릿
gof_patch/             GOF 배경제거 패치
jetson/                젯슨 촬영 코드
experiments/           VGGT 깊이 진단 (제품 경로 아님)
docs/                  작업 기록
```

계산부 파이썬 파일은 서로를 같은 폴더에서 import 하므로 **루트에 평평하게** 둔다.
하위 폴더로 옮기면 깨진다.

## 환경 변수

| 변수 | 용도 | 기본 |
|---|---|---|
| `VGGT_HOST` | 서버 바인드 주소 | `0.0.0.0` (젯슨 접속용) |
| `COLMAP_EXE` | COLMAP 실행 파일 | 몇 군데 탐색 |
| `GOF_DIR` | Gaussian Opacity Fields 위치 | `~/gaussian-opacity-fields` |
| `GOF_PYTHON` | GOF 확장이 설치된 파이썬 | VGGT 가상환경 |
| `SCANNER_DIR` | MuJoCo 모델 폴더 (`rig_poses.py`) | — |

⚠ 서버는 기본으로 `0.0.0.0` 에 바인드한다. PC 에 공인 IP 가 직접 붙어 있으면
5000 포트가 인터넷에 노출된다. 혼자 테스트할 때는 `set VGGT_HOST=127.0.0.1`.

## 요구 사항

- Windows 10/11, NVIDIA GPU (RTX 3070 8GB 에서 개발·검증)
- Python 3.10~3.12
- 정밀 경로: COLMAP (CUDA 빌드)
- 가우시안: Visual Studio 2022 빌드 도구 (CUDA 확장 컴파일)

패키지는 [`requirements.txt`](requirements.txt). 보통은 `VGGT-Server.bat` 이 알아서 설치한다.
