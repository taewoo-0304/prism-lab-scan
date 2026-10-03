"""
촬영 시퀀스 리그 (젯슨, 시스템 파이썬으로 실행 — cv2 필요)

정해진 좌표를 순서대로 돌면서 촬영하고, 각 사진에 대응하는 카메라 좌표를
poses.json 으로 남깁니다. 이 좌표가 있으면 노트북이 mm 스케일을 '추정'이 아니라
'확정'할 수 있고, 덤으로 스캔 품질(잔차)까지 자동으로 나옵니다.

모터가 아직 없어도 지금 쓸 수 있습니다:
  --rig manual  → 목표 좌표를 화면에 띄우고, 사람이 맞춘 뒤 엔터를 치면 촬영.
  좌표가 기록되므로 스케일 보정은 모터를 붙인 뒤와 똑같이 동작합니다.

모터를 붙일 때는 RigController를 상속해서 move_to() 하나만 구현하면 됩니다.

사용법:
  python3 rig.py <저장폴더> --radius 200 --heights 60,120,180 --n 12 --rig manual
"""
import os, sys, json, time, math, argparse

UP_AXIS = "z"          # 이 좌표계에서 위쪽은 +Z (물체 바닥면이 z=0)


# ---------- 좌표 시퀀스 ----------
def orbit_poses(radius_mm, heights_mm, n_per_ring, start_deg=0.0):
    """물체를 원점에 두고, 각 높이마다 반지름 radius_mm 원을 n_per_ring등분.
    좌표계: 오른손, +Z가 위, 물체 중심이 원점 (바닥 아님)."""
    poses = []
    i = 0
    for ring, hz in enumerate(heights_mm):
        for k in range(n_per_ring):
            deg = start_deg + 360.0 * k / n_per_ring
            r = math.radians(deg)
            poses.append({
                "i": i,
                "xyz": [round(radius_mm * math.cos(r), 2),
                        round(radius_mm * math.sin(r), 2),
                        round(float(hz), 2)],
                "deg": round(deg, 2),
                "ring": ring,
            })
            i += 1
    return poses


# ---------- 이동 제어 ----------
class RigController:
    """모터를 붙일 때 이 클래스를 상속해서 move_to()만 구현하세요."""
    name = "base"

    def move_to(self, pose):
        raise NotImplementedError

    def settle(self, sec=0.4):
        time.sleep(sec)          # 진동이 멎기를 기다림. 흔들리면 블러가 생김

    def close(self):
        pass


class ManualRig(RigController):
    """모터 없이 사람이 직접 맞추는 모드. 좌표는 그대로 기록됩니다."""
    name = "manual"

    def move_to(self, pose):
        x, y, z = pose["xyz"]
        print(f"\n  → [{pose['i']+1}] 각도 {pose['deg']:.0f}°  높이 {z:.0f}mm  "
              f"(x {x:.0f}, y {y:.0f})")
        input("     위치를 맞추고 엔터 (s=이 장 건너뛰기): ")


class DummyRig(RigController):
    """배선 전 흐름 확인용 — 이동 없이 그 자리에서 연속 촬영"""
    name = "dummy"

    def move_to(self, pose):
        pass

    def settle(self, sec=0.05):
        time.sleep(sec)


def make_rig(kind):
    return {"manual": ManualRig, "dummy": DummyRig}.get(kind, ManualRig)()


# ---------- 카메라 ----------
def open_camera(index=0, width=1280, height=720, warmup=15, log=print):
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(warmup):          # 자동 노출/화이트밸런스가 수렴하도록
        cap.read()

    # 수렴한 뒤 고정. 프레임마다 밝기·초점이 바뀌면 정합 품질이 떨어짐
    locked = []
    for prop, val, label in [(cv2.CAP_PROP_AUTOFOCUS, 0, "AF"),
                             (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "AE"),
                             (cv2.CAP_PROP_AUTO_WB, 0, "AWB")]:
        try:
            if cap.set(prop, val):
                locked.append(label)
        except Exception:
            pass
    log(f"  카메라 고정: {', '.join(locked) if locked else '(드라이버가 지원 안 함 — 확인 필요)'}")
    return cap


def grab_fresh(cap, flush=4):
    """cv2가 프레임을 버퍼링하므로, 이동 후엔 몇 장 버리고 최신 프레임을 받아야 함"""
    for _ in range(flush):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        ok, frame = cap.read()
    return ok, frame


# ---------- 시퀀스 촬영 ----------
def capture_sequence(folder, poses, rig, cam_index=0, settle=0.4, log=print):
    import cv2
    os.makedirs(folder, exist_ok=True)
    cap = open_camera(cam_index, log=log)
    taken = []
    try:
        for pose in poses:
            try:
                rig.move_to(pose)
            except KeyboardInterrupt:
                log("\n  (중단)")
                break
            rig.settle(settle)
            ok, frame = grab_fresh(cap)
            if not ok:
                log(f"    [{pose['i']}] 촬영 실패 — 건너뜀")
                continue
            p = os.path.join(folder, f"img_{len(taken):03d}.jpg")
            cv2.imwrite(p, frame)
            rec = dict(pose)
            rec["file"] = os.path.basename(p)
            rec["i"] = len(taken)          # 실제 저장 순서로 다시 매김
            taken.append(rec)
            log(f"    저장 {len(taken)}장  {rec['file']}")
    finally:
        cap.release()
        rig.close()

    write_poses(folder, taken, rig.name)
    return taken


def write_poses(folder, poses, rig_name="manual"):
    doc = {
        "unit": "mm",
        "frame": "rig",
        "up": UP_AXIS,
        "rig": rig_name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "poses": poses,
    }
    path = os.path.join(folder, "poses.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"  좌표 기록: {path}  ({len(poses)}장)")
    return path


def load_poses(folder):
    """(xyz 리스트, up축) — 없으면 (None, None)"""
    path = os.path.join(folder, "poses.json")
    if not os.path.exists(path):
        return None, None
    try:
        doc = json.load(open(path, encoding="utf-8"))
        return doc, doc.get("up", UP_AXIS)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--radius", type=float, default=200.0, help="궤도 반지름 mm")
    ap.add_argument("--heights", default="60,120,180", help="높이 mm, 쉼표 구분")
    ap.add_argument("--n", type=int, default=12, help="한 바퀴당 장수")
    ap.add_argument("--rig", default="manual", choices=["manual", "dummy"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.4)
    a = ap.parse_args()

    heights = [float(h) for h in a.heights.split(",") if h.strip()]
    poses = orbit_poses(a.radius, heights, a.n)
    folder = os.path.expanduser(a.folder)

    print(f"시퀀스: 반지름 {a.radius:.0f}mm · 높이 {heights} · 한 바퀴 {a.n}장 "
          f"→ 총 {len(poses)}장")
    print(f"저장: {folder}")
    if a.rig == "manual":
        print("모드: 수동 — 목표 위치를 맞추고 엔터. 거리를 일정하게 유지하세요.")
    capture_sequence(folder, poses, make_rig(a.rig), a.cam, a.settle)
    print("완료")


if __name__ == "__main__":
    main()
