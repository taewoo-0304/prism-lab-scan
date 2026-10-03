"""3DGS/GOF point_cloud.ply → .splatsh (구면조화 포함 웹 뷰어용).

.splat(32바이트/개)은 기본색만 담아서 금속 반사가 사라진다. render.py 결과와
나란히 보면 밋밋한 플라스틱처럼 보인다. 학습된 값 63개 중 45개가 '보는 각도에
따른 색 변화'(구면조화 3차)인데 그걸 통째로 버리기 때문이다.

여기서는 그 45개를 uint8 로 눌러 담는다. 실측 범위가 -0.577~0.607 로 좁아서
[-0.62, 0.62] 선형 양자화면 1/206 해상도가 나온다 — 눈에 안 띈다.

  파일 구조
    "SPLATSH1"        8B   매직
    count             4B   uint32
    sh_scale          4B   float32   (실제값 = (u8-128)/127 * sh_scale)
    dc_scale          4B   float32   (DC 는 범위가 넓어 따로 둔다)
    기본 블록         count x 32B    .splat 과 동일
    구면조화 블록     count x 45B    채널 우선 (R 계수 1..15, G, B)

용량은 32→77 바이트로 늘어 3.0MB → 7.2MB 가 된다.
"""
import os, struct, sys
import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814
SH_RANGE = 0.62
# DC(0번 계수)는 물체의 기본색이라 범위가 훨씬 넓다. 실측 -2.298 ~ 5.140.
# 이걸 SH_RANGE 로 같이 눌렀다가 값의 68.9%가 잘려서 렌더에 흰 섬광이 생겼다.
DC_RANGE = 5.2


def convert(ply_path, out_path, drop_needles=True):
    el = PlyData.read(ply_path)["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([el[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    rot = np.stack([el[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9
    opacity = 1.0 / (1.0 + np.exp(-el["opacity"]))
    dc = np.stack([el[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    rest = np.stack([el[f"f_rest_{i}"] for i in range(45)], 1).astype(np.float32)

    if drop_needles:
        mx, mn = scale.max(1), np.maximum(scale.min(1), 1e-9)
        bad = (mx / mn > 30) & (mx > np.percentile(mx, 97))
        keep = ~bad
        print(f"  바늘 제거 {int(bad.sum()):,}개 ({100*bad.mean():.1f}%)")
        xyz, scale, rot, opacity, dc, rest = (
            a[keep] for a in (xyz, scale, rot, opacity, dc, rest))

    n = len(xyz)
    order = np.argsort(-(scale.prod(1) * opacity))
    xyz, scale, rot, opacity, dc, rest = (
        a[order] for a in (xyz, scale, rot, opacity, dc, rest))

    base = np.zeros((n, 32), np.uint8)
    base[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    base[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
    # 색 자리에는 DC 계수를 그대로 넣는다. 셰이더가 0.5 + C0*dc 로 되돌린다.
    # (기존 .splat 과 달리 여기서 미리 색으로 바꾸면 구면조화를 더할 수 없다.)
    base[:, 24:27] = np.clip((dc / DC_RANGE * 127) + 128, 0, 255).astype(np.uint8)
    base[:, 27] = np.clip(opacity * 255, 0, 255).astype(np.uint8)
    base[:, 28:32] = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

    clipped = float((np.abs(rest) > SH_RANGE).mean())
    dc_clip = float((np.abs(dc) > DC_RANGE).mean())
    sh = np.clip((rest / SH_RANGE * 127) + 128, 0, 255).astype(np.uint8)

    with open(out_path, "wb") as f:
        f.write(b"SPLATSH1")
        f.write(struct.pack("<Iff", n, SH_RANGE, DC_RANGE))
        f.write(base.tobytes())
        f.write(sh.tobytes())

    print(f"{n:,}개 → {out_path}  ({os.path.getsize(out_path)/1e6:.1f}MB)")
    print(f"  구면조화 ±{SH_RANGE} 잘림 {100*clipped:.3f}% · "
          f"DC ±{DC_RANGE} 잘림 {100*dc_clip:.3f}%")
    return out_path


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
