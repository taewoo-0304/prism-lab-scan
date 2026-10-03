"""3DGS/GOF 의 point_cloud.ply → .splat (웹 뷰어용 32바이트/개 형식).

왜 변환하나 — 학습 결과 ply 는 가우시안 하나에 63개 값(구면조화 45개 포함)을
담아 24MB다. 웹에서 돌릴 땐 시점별 색 변화를 뺀 기본색만 있으면 충분해서
32바이트로 줄어든다(9.5만 개 → 3MB). 브라우저가 바로 읽는다.

  위치   float32 x3   12B
  크기   float32 x3   12B   (ply 는 log 스케일이라 exp 를 씌운다)
  색     uint8   x4    4B   (구면조화 DC 항 → RGB, 시그모이드(opacity) → A)
  회전   uint8   x4    4B   (쿼터니언을 128 기준으로 정규화)

정렬 — 뷰어는 매 프레임 깊이순으로 다시 정렬한다. 그래도 여기서 '큰 것 ×
불투명한 것' 순으로 미리 넣어두면 파일을 앞에서부터 읽는 동안 형체가
먼저 잡힌다.
"""
import sys, os
import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814          # 구면조화 0차 계수


def convert(ply_path, out_path, drop_needles=True):
    el = PlyData.read(ply_path)["vertex"]
    n = el.count
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([el[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    rot = np.stack([el[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9
    opacity = 1.0 / (1.0 + np.exp(-el["opacity"]))
    rgb = 0.5 + C0 * np.stack([el[f"f_dc_{i}"] for i in range(3)], 1)

    if drop_needles:
        # 3DGS 는 표면을 얇은 판으로 덮으려다 바늘처럼 가늘고 긴 가우시안을
        # 만든다. 학습 시점에서는 다른 것에 가려 안 보이지만, 자유롭게 돌리면
        # 물체 밖으로 뻗은 검은 가시가 된다. 실측(모터, 94,602개):
        #   가늘기(최대축/최소축) 분위 50/99/100 = 6.4 / 71 / 1632
        #   최대축 길이           분위 50/99/100 = 0.014 / 0.077 / 0.62
        #   (물체 전체 크기가 2.0 단위다 — 0.62 짜리 바늘은 절반을 가로지른다)
        mx, mn = scale.max(1), np.maximum(scale.min(1), 1e-9)
        bad = (mx / mn > 30) & (mx > np.percentile(mx, 97))
        keep = ~bad
        print(f"  바늘 제거 {int(bad.sum()):,}개 ({100*bad.mean():.1f}%)")
        xyz, scale, rot, opacity, rgb = (a[keep] for a in (xyz, scale, rot, opacity, rgb))
        n = len(xyz)

    order = np.argsort(-(scale.prod(1) * opacity))    # 큰 것·진한 것 먼저
    xyz, scale, rot, opacity, rgb = (a[order] for a in (xyz, scale, rot, opacity, rgb))

    buf = np.zeros((n, 32), np.uint8)
    buf[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
    buf[:, 24:27] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    buf[:, 27] = np.clip(opacity * 255, 0, 255).astype(np.uint8)
    buf[:, 28:32] = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    print(f"{n:,}개 → {out_path}  ({os.path.getsize(out_path)/1e6:.1f}MB)")
    print(f"  중심 {np.round(xyz.mean(0), 3)} · 크기 {np.round(xyz.max(0)-xyz.min(0), 3)}")
    return out_path


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
