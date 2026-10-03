# GOF 패치 — SAM 마스크를 학습 손실에 반영

[Gaussian Opacity Fields](https://github.com/autonomousvision/gaussian-opacity-fields)
원본 코드는 이 저장소에 넣지 않았다(그쪽 라이선스를 따른다). 대신 바꾼 부분만
패치로 둔다.

## 적용

```bash
git clone https://github.com/autonomousvision/gaussian-opacity-fields.git %USERPROFILE%\gaussian-opacity-fields
cd %USERPROFILE%\gaussian-opacity-fields
git checkout 5245b20e5d11acd6d1ff5af4b890dc2bedd99693
git apply <이 저장소>\gof_patch\gof_masked_loss.patch
```

검증: 위 커밋의 깨끗한 사본에 `git apply --check` 통과, 적용 결과가 개발 PC의
수정본과 내용 동일(줄바꿈 무시).

`gof_pipe.py` 는 `~/gaussian-opacity-fields` 를 기본으로 찾는다. 다른 곳에 두면
환경변수 `GOF_DIR` 로 지정한다.

## 왜 필요한가

`gof_prep.py --sam` 이 마스크를 이미지 알파로 구워 넣어도, 원본 GOF 는 그 알파를
**학습에 쓰지 않는다.**

| 파일 | `gt_alpha_mask` 사용 |
|---|---|
| `utils/camera_utils.py` | RGBA 에서 알파를 떼어 저장 |
| `scene/cameras.py:44` | `original_image *= gt_alpha_mask` 가 **주석 처리돼 있음** |
| `train.py` | **한 번도 안 씀** |
| `extract_mesh_tsdf.py` | 여기서만 씀 |

그래서 배경을 지운 점군으로 시작해도 학습이 사진 속 방을 보고 배경 가우시안을
다시 만든다. 실측(모터, 15000 iter): 렌더에 책상·케이블·벽이 그대로 나왔다.

## 무엇을 바꿨나

**`train.py`**

1. GT 의 배경을 렌더 배경색으로 덮는다. 렌더도 같은 색 위에 합성되므로
   "물체 밖에서는 알파가 0" 이 그대로 RGB 손실이 된다. GT 만 검게 하면 경계에
   검은 테가 끼지만, 양쪽 배경색이 같으니 그 문제는 없다.
2. 누적 알파(렌더 채널 7)를 마스크에 직접 맞추는 손실을 더한다. 1번만으로는
   배경에 뜬 가우시안이 배경색과 같은 색이면 손실이 0 이라 살아남는다.
3. `distortion_loss`·`depth_normal_loss` 를 마스크 안에서만 평균낸다. 배경 픽셀을
   넣으면 정규화가 지우려는 그 가우시안을 다듬는 데 쓰인다.
4. `training_report` 의 PSNR 도 같은 기준으로 잰다. 안 고치면 학습이 잘될수록
   PSNR 이 떨어진다(실측: 200 iter 에서 4.9 로 표시됐다).

**`arguments/__init__.py`**

- `lambda_mask = 1.0` 추가 → `--lambda_mask` 로 조절. **0 이면 원본과 동일하게 동작한다.**

## 결과 (모터, 113장, 30000 iter, 43분)

| iter | PSNR |
|---|---|
| 7000 | 28.07 |
| **15000** | **29.06** |
| 30000 | 28.63 |

배경 픽셀 밝기 0.0. 15000 이 30000 보다 낫다 — 그 뒤는 과적합.
스모크 테스트에서는 2000 iter 만에 PSNR 26.6, 배경 완전 제거.
