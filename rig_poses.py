"""manifest.json(촬영 기록) → poses.json(카메라 mm 좌표).

왜 필요한가
    VGGT도 COLMAP도 스케일을 모른다. 사진만 보고는 5cm짜리인지 5m짜리인지 알 수
    없어서 결과가 단위 없는 숫자로 나온다. 리그가 카메라를 **어디에 두고 찍었는지**
    알려주면 그때 비로소 mm가 붙고, 덤으로 도면 up축이 자동으로 정해지고
    복원 궤적과 실제 궤적의 잔차가 품질 지표로 나온다.

왜 손으로 계산하지 않는가
    scan_multi.py 가 다중 링에서 poses.json 을 일부러 안 만든다. camera_xyz() 가
    반경·높이 상수인 평면 원이라 R이 움직이면 틀린 좌표가 되기 때문이고,
    그 판단은 옳다 — 틀린 좌표는 없는 것보다 나쁘다. VGGT/COLMAP이 그걸로
    스케일을 확정해버려서 치수가 통째로 틀어진다.
    그래서 여기서는 검증된 MuJoCo 모델(scanner.xml)의 정기구학을 쓴다.
    2026-08-13 작업 기록에 모델 메시 실측이 문서값과 일치함이 확인돼 있다
    (턴테이블 반경 90.0mm, 상면 +79.0mm).
    ⚠ motor.py 의 헤드 위치 계산식은 쓰면 안 된다. 같은 기록에서 모델과 어긋남이
      확인됐다 (R=90°에서 모델 6.8mm vs motor.py -90mm).

안전장치
    쓰기 전에 반드시 self_check() 가 문서 실측값과 대조한다. 어긋나면 좌표를
    만들지 않고 예외를 던진다. "틀린 좌표를 쓰느니 없는 게 낫다"를 코드로 강제한다.

사용법
    python rig_poses.py <촬영폴더>            # manifest.json 을 읽어 poses.json 생성
    python rig_poses.py --self-check          # 기구학만 검증
"""
import os, sys, json, math, argparse
import numpy as np

# scan_multi.py 주석에 적힌 기준값.
#
# ⚠ 미검증이다. 이 숫자를 계산한 코드가 프로젝트에 없고(주석에만 존재),
#   같은 작업 기록이 남긴 다른 수치는 2026-08-16에 틀린 것으로 확인됐다
#   ("슬래브 두께가 154mm→6mm로 줄었다" → 실측하니 1.2배 차이였다).
#   그래서 여기 XML 기구학과 어긋날 때 어느 쪽이 맞는지 파일만으로는 못 정한다.
#
#   2026-08-16 현재 상태: XML 기구학과 아래 값이 불일치한다.
#     계산(z=0): R=55 → 방위 117.9°, 수평 135.3mm, 높이 233.9mm
#     주석      : R=55 → 방위 100.2°, 수평 113.5mm, 높이 223.1mm
#   높이 차는 Z 캐리지 위치(-13mm쯤)로 설명되지만 수평거리는 설명되지 않는다.
#   모델대로면 카메라가 T2축에서 63mm 떨어져 있고, 주석대로면 18mm다.
#
#   해결법 — 자로 한 번 재면 끝난다:
#     R을 90°(수직)에 두고, 턴테이블 회전축에서 렌즈까지의 수평거리(mm).
#     그 값을 --measured-h90 으로 주면 어느 쪽이 맞는지 판정한다.
REFERENCE = [
    # (모터 R각도, 방위각deg, 수평거리mm, 높이mm)
    (55.0,  100.2, 113.5, 223.1),
    (125.0, 259.8, 114.1, 222.7),
]
TOL_DEG, TOL_MM = 3.0, 8.0

_MODEL = {"m": None, "d": None, "site": None}


def _scanner_dir():
    for c in [os.environ.get("SCANNER_DIR", ""),
              os.path.join(os.path.expanduser("~"), "Desktop", "스캐너_전달",
                           "한이음_분광기스캐너제어")]:
        if c and os.path.isdir(os.path.join(c, "mujoco_model")):
            return c
    raise RuntimeError("스캐너 프로젝트 폴더를 못 찾았습니다. 환경변수 SCANNER_DIR 로 지정하세요")


def _load():
    if _MODEL["m"] is not None:
        return _MODEL
    import mujoco
    sys.path.insert(0, os.path.join(_scanner_dir(), "mujoco_model"))
    from model_loader import load_model
    m = load_model()
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "camera_view")
    if sid < 0:
        raise RuntimeError("scanner.xml 에 camera_view 사이트가 없습니다")
    _MODEL.update(m=m, d=d, site=sid, mujoco=mujoco)
    return _MODEL


def _qadr(m, mujoco, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(m.jnt_qposadr[j]) if j >= 0 else None


def camera_xyz(r_deg, plate_deg, z_m=0.0, x_deg=0.0, t1_deg=0.0):
    """모터 각도 → 카메라 위치 (mm, 턴테이블 축 원점, +Z 위).

    r_deg   : 모터 R 각도 (90 = 수직). 모델 관절각 = r_deg - 90
    plate_deg: 판(T2) 각도
    z_m     : Z 캐리지 (m, 0 = 최상단, 음수로 내려감)
    """
    M = _load(); m, d, mj = M["m"], M["d"], M["mujoco"]
    d.qpos[:] = 0
    for name, val in (("t2_joint", math.radians(plate_deg)),
                      ("t1_joint", math.radians(t1_deg)),
                      ("r_joint",  math.radians(r_deg - 90.0)),
                      ("z_joint",  z_m),
                      ("x_joint",  math.radians(x_deg))):
        a = _qadr(m, mj, name)
        if a is not None:
            lo, hi = m.jnt_range[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)]
            if hi > lo:
                val = float(np.clip(val, lo, hi))
            d.qpos[a] = val
    mj.mj_forward(m, d)
    return np.array(d.site_xpos[M["site"]], dtype=float) * 1000.0


def self_check(verbose=True):
    """문서 실측값과 대조. 통과 못 하면 좌표를 만들면 안 된다."""
    ok = True
    rows = []
    for r_deg, az_ref, hor_ref, h_ref in REFERENCE:
        p = camera_xyz(r_deg, 0.0)
        az = math.degrees(math.atan2(p[1], p[0])) % 360.0
        hor = math.hypot(p[0], p[1])
        d_az = min(abs(az - az_ref), 360 - abs(az - az_ref))
        good = (d_az <= TOL_DEG and abs(hor - hor_ref) <= TOL_MM
                and abs(p[2] - h_ref) <= TOL_MM)
        ok &= good
        rows.append((r_deg, az, az_ref, hor, hor_ref, p[2], h_ref, good))
    if verbose:
        print("기구학 검증 (문서 실측값 대조)")
        print("  R      방위각(계산/문서)      수평거리(계산/문서)    높이(계산/문서)")
        for r, a, ar, ho, hor_, z, zr, g in rows:
            print("  %5.0f  %6.1f / %6.1f     %6.1f / %6.1f mm   %6.1f / %6.1f mm  %s"
                  % (r, a, ar, ho, hor_, z, zr, "OK" if g else "불일치"))
        print("  →", "통과" if ok else "실패 — 좌표를 만들면 안 됩니다")
    return ok


def from_manifest(folder, z_m=0.0, x_deg=0.0, strict=True, log=print):
    """촬영 폴더의 manifest.json → poses.json.

    manifest 에는 지령값이 아니라 **엔코더 실측** r_deg/plate_deg 가 들어 있다.
    (scan_multi.py 가 절대위치 추종 대신 상대이동 후 도달값을 기록한다.)
    """
    mf = os.path.join(folder, "manifest.json")
    if not os.path.exists(mf):
        raise RuntimeError(f"manifest.json 이 없습니다: {folder}")
    doc = json.load(open(mf, encoding="utf-8"))
    shots = [s for s in doc.get("shots", []) if s.get("file")]
    if not shots:
        raise RuntimeError("manifest 에 촬영 기록이 없습니다")

    if not self_check(verbose=True):
        raise RuntimeError("기구학 검증 실패 — poses.json 을 만들지 않습니다. "
                           "틀린 좌표는 없는 것보다 나쁩니다")

    poses = []
    for s in shots:
        p = camera_xyz(float(s["r_deg"]), float(s["plate_deg"]), z_m, x_deg)
        poses.append({"i": len(poses), "file": s["file"],
                      "xyz": [round(float(v), 2) for v in p],
                      "r_deg": s["r_deg"], "plate_deg": s["plate_deg"]})
    out = {"unit": "mm", "frame": "rig", "up": "z",
           "source": "mujoco_fk(scanner.xml)",
           "note": "manifest.json 의 엔코더 실측 각도를 MuJoCo 정기구학으로 변환",
           "z_m": z_m, "x_deg": x_deg, "poses": poses}
    path = os.path.join(folder, "poses.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    A = np.array([p["xyz"] for p in poses])
    log(f"poses.json 저장 {len(poses)}개 → {path}")
    log(f"  궤도 반지름 중앙값 {np.median(np.hypot(A[:,0],A[:,1])):.1f}mm · "
        f"높이 {A[:,2].min():.1f}~{A[:,2].max():.1f}mm")
    return path


def main():
    ap = argparse.ArgumentParser(description="촬영 기록 → 카메라 좌표(mm)")
    ap.add_argument("folder", nargs="?", help="manifest.json 이 있는 촬영 폴더")
    ap.add_argument("--self-check", action="store_true", help="기구학만 검증")
    ap.add_argument("--z", type=float, default=0.0, help="Z 캐리지 위치(m, 0=최상단)")
    ap.add_argument("--x", type=float, default=0.0, help="X 각도(deg)")
    ap.add_argument("--measured-h90", type=float, default=None,
                    help="R=90°일 때 T2축~렌즈 수평거리 실측(mm). 모델 검증용")
    a = ap.parse_args()
    if a.measured_h90 is not None:
        p90 = camera_xyz(90.0, 0.0, a.z, a.x)
        model_h = math.hypot(p90[0], p90[1])
        print("R=90° 수평거리   모델 %.1fmm  vs  실측 %.1fmm  (차이 %.1fmm)"
              % (model_h, a.measured_h90, abs(model_h - a.measured_h90)))
        print("  →", "모델이 맞습니다. REFERENCE 주석값을 버리세요"
              if abs(model_h - a.measured_h90) <= TOL_MM else
              "모델과 실측이 다릅니다. scanner.xml 의 치수를 점검해야 합니다")
        return 0
    if a.self_check or not a.folder:
        return 0 if self_check() else 1
    try:
        from_manifest(a.folder, a.z, a.x)
    except Exception as e:
        print(f"[실패] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
