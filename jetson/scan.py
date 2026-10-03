"""
통합 스캐너 v2 — 노트북 전용 (젯슨은 촬영/표시만)
  촬영 → 노트북 전송 → 3D 계산·정리·mm스케일(전부 노트북) → 3D 표시 → 도면

v1 대비 바뀐 것
  * 젯슨 로컬 계산(chain2) 경로 제거. 계산은 무조건 노트북
  * 노트북을 못 찾으면 계산 대신 재시도 (조용히 느린 젯슨으로 안 떨어짐)
  * 해상도/장수 제한 없음 — 서버가 알아서 최대로 돌림
  * 바닥/노이즈/최대거리/스케일 옵션을 노트북으로 전달

사용법:
  source ~/vggt_env/bin/activate
  python3 scan.py
"""
import os, sys, io, json, time, glob, base64, socket, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PORT = 5000
T_HEALTH = 1.0
T_CALC = 7200                        # 518 해상도 + 많은 장수는 오래 걸림
SYS_PY = "/usr/bin/python3"          # cv2가 있는 시스템 파이썬 (촬영용)
CFG = os.path.expanduser("~/vggt/offload_config.json")

# 촬영 코드 (시스템 파이썬에서 subprocess로 실행 — cv2 사용)
CAPTURE_CODE = r'''
import cv2, os, sys
folder = sys.argv[1]
os.makedirs(folder, exist_ok=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA_FAIL"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
n = 0
print("SPACE=촬영, q=종료")
while True:
    ret, frame = cap.read()
    if not ret: break
    disp = frame.copy()
    cv2.putText(disp, f"{n} shots  [SPACE]capture [q]done", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("capture", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        p = os.path.join(folder, f"img_{n:03d}.jpg")
        cv2.imwrite(p, frame); n += 1
        print(f"SAVED {n}")
    elif k == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
print(f"DONE {n}")
'''


# ---------- 노트북 탐색 ----------
def load_cfg():
    if os.path.exists(CFG):
        try: return json.load(open(CFG))
        except Exception: pass
    return {}

def save_cfg(c):
    os.makedirs(os.path.dirname(CFG), exist_ok=True)
    json.dump(c, open(CFG, "w"), indent=2)

def my_subnets():
    nets = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        nets.append(".".join(ip.split(".")[:3]))
    except Exception: pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout
        for ip in out.split():
            if ip.count(".") == 3 and not ip.startswith("127."):
                pre = ".".join(ip.split(".")[:3])
                if pre not in nets: nets.append(pre)
    except Exception: pass
    return nets

def probe(ip, timeout=None):
    import requests
    try:
        r = requests.get(f"http://{ip}:{PORT}/health", timeout=timeout or T_HEALTH)
        if r.status_code == 200:
            j = r.json()
            if j.get("alive"): j["_ip"] = ip; return j
    except Exception: pass
    return None

def find_laptop():
    cfg = load_cfg()
    if cfg.get("laptop_ip"):
        h = probe(cfg["laptop_ip"])
        if h: return h
    for net in my_subnets():
        print(f"  {net}.1~254 스캔...")
        with ThreadPoolExecutor(max_workers=64) as ex:
            for h in ex.map(probe, [f"{net}.{i}" for i in range(1, 255)]):
                if h:
                    cfg["laptop_ip"] = h["_ip"]; save_cfg(cfg); return h
    return None


def require_laptop():
    """노트북을 찾을 때까지. 계산은 무조건 노트북이므로 여기서 물러서지 않음."""
    while True:
        h = find_laptop()
        if h:
            return h
        print("\n  [!] 노트북을 못 찾았습니다. 확인할 것:")
        print("      1) 노트북에서 run_server.bat 이 떠 있는지")
        print("      2) 젯슨과 같은 공유기(같은 대역)에 붙어 있는지")
        print("      3) 윈도우 방화벽에서 5000 포트 인바운드 허용")
        a = input("  엔터=다시 찾기 / IP 직접 입력 / q=종료 : ").strip()
        if a.lower() == "q":
            return None
        if a:
            h = probe(a, timeout=8)
            if h:
                cfg = load_cfg(); cfg["laptop_ip"] = a; save_cfg(cfg)
                return h
            print(f"  {a} 응답 없음")


# ---------- 촬영 ----------
def capture(folder):
    print("\n[촬영] 카메라 창에서 SPACE=촬영, q=종료")
    print("  물체 고정, 카메라 들고 한 바퀴, 시작 거리 일정하게")
    print("  ※ 노트북이 계산하므로 장수 제한 없습니다. 20~40장이면 품질이 확 올라갑니다")
    subprocess.run([SYS_PY, "-c", CAPTURE_CODE, folder])
    imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    print(f"  촬영 {len(imgs)}장")
    return imgs


# ---------- 원격 계산 ----------
def offload(ip, paths, **opt):
    import requests
    files = [("images", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in paths]
    data = {k: str(v) for k, v in opt.items()}
    t0 = time.time()
    try:
        r = requests.post(f"http://{ip}:{PORT}/reconstruct",
                          files=files, data=data, timeout=T_CALC)
    finally:
        for _, (_, f, _) in files:
            f.close()
    if r.status_code != 200:
        try:
            j = r.json()
            for line in j.get("log", [])[-10:]:
                print(f"    | {line}")
            raise RuntimeError(j.get("error", r.text[:300]))
        except ValueError:
            raise RuntimeError(f"서버 오류: {r.text[:300]}")
    j = r.json()
    for line in j.get("log", []):
        print(f"    | {line}")
    z = np.load(io.BytesIO(base64.b64decode(j["npz_b64"])))
    P, C = z["points"], z["colors"]
    print(f"  점 {len(P)} · 서버 {j['elapsed_sec']}s · 왕복 {time.time()-t0:.1f}s "
          f"· {j['n_images']}장 @ {j['res']} (묶음 {j.get('group')})")
    src = j.get("scale_source", "none")
    if j.get("dims_mm"):
        d = j["dims_mm"]
        label = {"poses": "리그 좌표 기준", "orbit": "궤도반지름 추정",
                 "size": "물체크기 추정"}.get(src, src)
        print(f"  실측(mm): X {d[0]:.1f} · Y {d[1]:.1f} · Z {d[2]:.1f}   [{label}]")
    elif float(data.get("ref_mm", 0)) > 0 or "poses_json" in data:
        print("  [!] 스케일 보정 실패 — 도면 치수는 참고용입니다")

    if src == "poses" and j.get("fit_rms_mm") is not None:
        rms, mx = j["fit_rms_mm"], j["fit_max_mm"]
        print(f"  [품질] 궤적 잔차 RMS {rms}mm / 최대 {mx}mm", end="")
        print("  ← 작을수록 신뢰도 높음" if rms < 5 else "  ← 큽니다. 재촬영을 권합니다")
    return P, C, j


def save_and_show(P, C, out):
    import trimesh
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if C is None or len(C) != len(P):
        C = np.full((len(P), 3), 180, np.uint8)
    trimesh.PointCloud(vertices=P, colors=C.astype(np.uint8)).export(out)
    print(f"  저장: {out}")
    subprocess.Popen(["meshlab", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


RIG_PY = os.path.join(HERE, "rig.py")


def capture_sequence(folder):
    """rig.py로 정해진 좌표를 돌며 촬영 (poses.json 생성)"""
    print("\n[촬영 — 시퀀스]")
    r_raw = input("  궤도 반지름 mm (기본 200): ").strip()
    h_raw = input("  높이들 mm, 쉼표 (기본 60,120,180): ").strip()
    n_raw = input("  한 바퀴당 장수 (기본 12): ").strip()
    mode = input("  이동 방식 manual/dummy (기본 manual): ").strip().lower() or "manual"
    cmd = [SYS_PY, RIG_PY, folder,
           "--radius", r_raw or "200",
           "--heights", h_raw or "60,120,180",
           "--n", n_raw or "12",
           "--rig", mode if mode in ("manual", "dummy") else "manual"]
    subprocess.run(cmd)
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def read_poses(folder):
    p = os.path.join(folder, "poses.json")
    if not os.path.exists(p):
        return None, None
    try:
        doc = json.load(open(p, encoding="utf-8"))
        return json.dumps(doc), doc.get("up", "z")
    except Exception as e:
        print(f"  [!] poses.json 읽기 실패: {e}")
        return None, None


# ---------- 스캔 1회 ----------
def one_scan(h):
    stamp = time.strftime("%m%d_%H%M%S")

    # ----- 사진 -----
    print("\n[1/3] 사진")
    print("  1) 시퀀스 촬영 (좌표 기록 → mm 치수 정확, 권장)")
    print("  2) 수동 촬영 (SPACE/q)")
    print("  3) 기존 폴더 사용")
    sel = input("  선택 (기본 1): ").strip() or "1"

    if sel == "3":
        folder = os.path.expanduser(input("  폴더 경로: ").strip())
        paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                       glob.glob(os.path.join(folder, "*.png")))
        if not paths:
            print(f"  [!] {folder} 에 사진이 없습니다. 이번 스캔 취소")
            return
        print(f"  기존 사진 {len(paths)}장 사용: {folder}")
    else:
        folder = os.path.expanduser(f"~/vggt/scan_{stamp}")
        paths = capture_sequence(folder) if sel == "1" else capture(folder)
        if len(paths) < 3:
            print("  [!] 사진이 너무 적음. 이번 스캔 취소")
            return

    poses_json, up_from_rig = read_poses(folder)
    if poses_json:
        print(f"  좌표 {len(json.loads(poses_json)['poses'])}개 확인 "
              f"→ mm 스케일을 좌표로 확정합니다")

    # ----- 설정 -----
    print("\n[설정]  (엔터 = 기본값)")
    name = input("  물체 이름 (도면용, 기본 '물체'): ").strip() or "물체"

    if poses_json:
        ref_mm, up = 0.0, (up_from_rig or "z")
        print(f"  카메라 거리·위쪽 축 질문 생략 (좌표에서 자동, up={up})")
    else:
        cam_raw = input("  카메라~물체 거리 cm (도면 mm 치수용, 엔터=생략): ").strip()
        ref_mm = float(cam_raw) * 10 if cam_raw else 0.0
        up = input("  위쪽 축 x/y/z (기본 y): ").strip().lower() or "y"

    sam_prompt = ""
    if h.get("sam"):
        sam_prompt = input("  SAM 물체 설명 (예: 'the object on the table', "
                           "엔터=안 씀): ").strip()
    elif h.get("sam_reason"):
        print(f"  (SAM 사용 불가: {h['sam_reason']})")

    md_raw = input("  이 거리(cm) 넘는 배경 자르기 (엔터=안 함): ").strip()
    max_dist_mm = float(md_raw) * 10 if md_raw else 0.0
    if max_dist_mm > 0 and ref_mm <= 0 and not poses_json:
        print("  [!] 배경 자르기는 카메라 거리나 좌표가 있어야 합니다 → 끔")
        max_dist_mm = 0.0

    fl_raw = input("  바닥 평면 제거 두께 (기본 15, 0=안 함): ").strip()
    floor = float(fl_raw) if fl_raw else 15.0

    stl_raw = input("  STL(3D프린팅용)도 만들까요? y/N: ").strip().lower()
    make_stl = (stl_raw == "y")

    # ----- 노트북 계산 -----
    print(f"\n[2/3] 3D 계산 (노트북 {h['_ip']})")
    out = os.path.expanduser("~/vggt/result_chain.ply")
    opt = dict(res=h.get("max_res", 518),
               conf_pct=25, dist_pct=50, overlap=2,
               ref_mm=ref_mm, ref_mode="d",
               max_dist_mm=max_dist_mm, floor=floor, noise_std=1.5)
    if poses_json:
        opt["poses_json"] = poses_json
    if sam_prompt:
        opt["sam_prompt"] = sam_prompt
    P, C, meta = offload(h["_ip"], paths, **opt)
    if meta.get("up"):
        up = meta["up"]
    save_and_show(P, C, out)

    # ----- 도면 -----
    print("\n[3/3] 도면 생성")
    pdf = None
    try:
        import make_drawing
        pdf_path = os.path.expanduser(f"~/vggt/도면_{stamp}.pdf")
        stl_path = os.path.expanduser(f"~/vggt/model_{stamp}.stl")
        pdf = make_drawing.make_drawing(out, name=name, up=up,
                                        out_pdf=pdf_path, out_stl=stl_path,
                                        make_stl=make_stl)
        if pdf and os.path.exists(pdf):
            subprocess.Popen(["xdg-open", pdf],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  [도면 실패] {e}")

    try:
        import shutil
        shutil.copy(out, os.path.join(folder, f"result_{stamp}.ply"))
    except Exception:
        pass

    print("\n완료!")
    print(f"  사진/결과 폴더: {folder}")
    print(f"  점군: {out}")
    if pdf:
        print(f"  도면: {pdf}")
    if make_stl:
        print(f"  STL : {os.path.expanduser(f'~/vggt/model_{stamp}.stl')}")


def main():
    print("=" * 55)
    print("  통합 3D 스캐너 v2 — 계산은 노트북")
    print("=" * 55)

    print("\n[부팅] 노트북 탐색 중...")
    h = require_laptop()
    if not h:
        print("노트북 없이는 계산할 수 없습니다. 종료합니다.")
        return

    prof = h.get("profile", {})
    print(f"  노트북: {h['_ip']}  장치={h['device']}  GPU={h.get('gpu')} "
          f"({h.get('vram_gb')}GB)")
    if h.get("version", 1) < 2:
        print("  [!] 노트북 server.py가 구버전입니다(v1).")
        print(f"      장수 {prof.get('max_imgs')}장 · 해상도 {prof.get('res')} 로 묶여 있고,")
        print("      바닥/노이즈 제거와 스케일이 동작하지 않습니다.")
        print("      노트북의 server.py를 v2로 교체하세요.")
    else:
        print(f"  제한: 장수 무제한 · 해상도 최대 {h.get('max_res')} "
              f"(자동 묶음 {h.get('auto_group')}장)")
        print(f"  SAM 3: {'사용 가능' if h.get('sam') else '없음'}"
              f"{'' if h.get('sam') else ' — ' + str(h.get('sam_reason'))}")
    if h.get("device") != "cuda":
        print("  [!] 노트북이 CPU 모드입니다 — 매우 느립니다. 노트북 CUDA 설치를 확인하세요.")

    print("=" * 55)

    scan_no = 0
    while True:
        scan_no += 1
        print("\n" + "=" * 55)
        print(f"  스캔 #{scan_no}")
        print("=" * 55)
        try:
            one_scan(h)
        except KeyboardInterrupt:
            print("\n  (중단됨)")
        except Exception as e:
            print(f"  [오류] {e}")
            if not probe(h["_ip"]):
                print("  노트북 응답 없음 → 다시 탐색")
                h2 = require_laptop()
                if not h2:
                    break
                h = h2

        print("\n" + "-" * 55)
        again = input("다시 스캔할까요?  엔터=계속 / q=종료 : ").strip().lower()
        if again == "q":
            print("종료합니다.")
            break
        print("  ※ MeshLab/PDF 창은 닫아주세요 (메모리 확보)")


if __name__ == "__main__":
    main()
