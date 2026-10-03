"""
젯슨 클라이언트 (단독 CLI) — 사진 폴더를 노트북으로 넘겨서 계산
  * 계산은 무조건 노트북. 젯슨 로컬 폴백 없음
  * 해상도/장수 제한 없음 (서버가 알아서 최대로)
사용법:
  python3 offload.py <사진폴더>                    ← IP 몰라도 됨
  python3 offload.py <사진폴더> <노트북IP>         ← IP 직접 지정
  python3 offload.py <사진폴더> --mm 150 --floor 15 --maxdist 300
     --mm      카메라~물체 거리(mm). 주면 결과가 mm 단위로 나옴
     --floor   바닥 평면 제거 두께 (기본 15, 0=안 함)
     --maxdist 이 거리(mm) 넘는 배경 제거 (--mm 필요)
     --res     해상도 직접 지정 (기본: 서버 최대 = 518)
* 평소에는 scan.py 하나만 쓰면 됩니다. 이건 이미 찍어둔 폴더를 재계산할 때 용도.
"""
import os, sys, io, json, time, base64, glob, subprocess, socket
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CFG = os.path.expanduser("~/vggt/offload_config.json")
PORT = 5000
T_HEALTH = 1.0
T_CALC = 7200


def load_cfg():
    if os.path.exists(CFG):
        try:
            return json.load(open(CFG))
        except Exception:
            pass
    return {}


def save_cfg(c):
    os.makedirs(os.path.dirname(CFG), exist_ok=True)
    json.dump(c, open(CFG, "w"), indent=2)


def my_subnets():
    nets = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        nets.append(".".join(ip.split(".")[:3]))
    except Exception:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout
        for ip in out.split():
            if ip.count(".") == 3 and not ip.startswith("127."):
                pre = ".".join(ip.split(".")[:3])
                if pre not in nets:
                    nets.append(pre)
    except Exception:
        pass
    return nets


def probe(ip, timeout=None, verbose=False):
    import requests
    try:
        r = requests.get(f"http://{ip}:{PORT}/health", timeout=timeout or T_HEALTH)
        if r.status_code == 200:
            j = r.json()
            if j.get("alive"):
                j["_ip"] = ip
                return j
    except Exception as e:
        if verbose:
            print(f"    연결 실패: {type(e).__name__} — {str(e)[:120]}")
    return None


def find_laptop(verbose=True):
    cfg = load_cfg()
    if cfg.get("laptop_ip"):
        if verbose:
            print(f"  저장된 IP 확인: {cfg['laptop_ip']}")
        h = probe(cfg["laptop_ip"])
        if h:
            return h
        if verbose:
            print("    응답 없음 → 네트워크 재탐색")
    for net in my_subnets():
        if verbose:
            print(f"  {net}.1~254 스캔 중...")
        ips = [f"{net}.{i}" for i in range(1, 255)]
        with ThreadPoolExecutor(max_workers=64) as ex:
            for h in ex.map(probe, ips):
                if h:
                    if verbose:
                        print(f"    찾음! {h['_ip']}")
                    cfg = load_cfg(); cfg["laptop_ip"] = h["_ip"]; save_cfg(cfg)
                    return h
    return None


def offload(ip, paths, **opt):
    import requests
    files = [("images", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in paths]
    data = {k: str(v) for k, v in opt.items() if v is not None}
    t0 = time.time()
    try:
        r = requests.post(f"http://{ip}:{PORT}/reconstruct",
                          files=files, data=data, timeout=T_CALC)
    finally:
        for _, (_, f, _) in files:
            f.close()
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"서버 오류: {r.text[:300]}")
    for line in j.get("log", []):
        print(f"    | {line}")
    if r.status_code != 200:
        raise RuntimeError(j.get("error", "알 수 없는 서버 오류"))
    z = np.load(io.BytesIO(base64.b64decode(j["npz_b64"])))
    P, C = z["points"], z["colors"]
    print(f"  점 {len(P)} · 서버 {j['elapsed_sec']}s · 왕복 {time.time()-t0:.1f}s "
          f"· {j['n_images']}장 @ {j['res']} (묶음 {j.get('group')})")
    if j.get("dims_mm"):
        d = j["dims_mm"]
        print(f"  실측(mm): X {d[0]:.1f} · Y {d[1]:.1f} · Z {d[2]:.1f}   "
              f"[{j.get('scale_source')}]")
    if j.get("fit_rms_mm") is not None:
        print(f"  [품질] 궤적 잔차 RMS {j['fit_rms_mm']}mm / 최대 {j['fit_max_mm']}mm")
    return P, C


def save_ply(P, C, out):
    import trimesh
    os.makedirs(os.path.dirname(out), exist_ok=True)
    trimesh.PointCloud(vertices=P, colors=C).export(out)
    print(f"  저장: {out}")
    subprocess.Popen(["meshlab", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args(argv):
    ip = None
    opt = {"ref_mm": 0, "ref_mode": "d", "floor": 15, "noise_std": 1.5,
           "max_dist_mm": 0, "conf_pct": 25, "dist_pct": 50, "res": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mm":
            opt["ref_mm"] = float(argv[i + 1]); i += 2
        elif a == "--floor":
            opt["floor"] = float(argv[i + 1]); i += 2
        elif a == "--maxdist":
            opt["max_dist_mm"] = float(argv[i + 1]); i += 2
        elif a == "--res":
            opt["res"] = int(argv[i + 1]); i += 2
        elif a == "--noise":
            opt["noise_std"] = float(argv[i + 1]); i += 2
        elif a == "--sam":
            opt["sam_prompt"] = argv[i + 1]; i += 2
        elif not a.startswith("--"):
            ip = a; i += 1
        else:
            print(f"[!] 모르는 옵션: {a}"); sys.exit(1)
    return ip, opt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = os.path.expanduser(sys.argv[1])
    ip_arg, opt = parse_args(sys.argv[2:])

    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        print(f"[!] {folder} 에 이미지 없음")
        sys.exit(1)
    print(f"사진 {len(paths)}장: {folder}")

    # 폴더에 poses.json이 있으면 자동으로 씀 (rig.py로 찍은 폴더)
    pj = os.path.join(folder, "poses.json")
    if os.path.exists(pj):
        try:
            doc = json.load(open(pj, encoding="utf-8"))
            if len(doc.get("poses", [])) == len(paths):
                opt["poses_json"] = json.dumps(doc)
                print(f"  poses.json 사용 ({len(doc['poses'])}개) → mm 스케일 확정")
            else:
                print(f"  [!] poses.json {len(doc.get('poses', []))}개 ≠ 사진 {len(paths)}장 → 무시")
        except Exception as e:
            print(f"  [!] poses.json 읽기 실패: {e}")

    if opt["max_dist_mm"] > 0 and opt["ref_mm"] <= 0 and "poses_json" not in opt:
        print("[!] --maxdist 는 --mm 이 있어야 동작합니다 → 끔")
        opt["max_dist_mm"] = 0

    print("\n[1] 노트북 찾는 중...")
    if ip_arg:
        print(f"  IP 직접 지정: {ip_arg} (최대 8초 대기)")
        h = probe(ip_arg, timeout=8, verbose=True)
        if h:
            cfg = load_cfg(); cfg["laptop_ip"] = ip_arg; save_cfg(cfg)
    else:
        h = find_laptop()

    if not h:
        print("\n[!] 노트북을 못 찾았습니다. 계산은 노트북에서만 합니다.")
        print("    - 노트북에서 run_server.bat 실행됐는지")
        print("    - 같은 공유기에 붙어 있는지")
        print("    - 윈도우 방화벽 5000 포트 인바운드 허용")
        sys.exit(1)

    ip = h["_ip"]
    print(f"\n[2] 노트북: {ip}  장치={h['device']}  GPU={h.get('gpu')} ({h.get('vram_gb')}GB)")
    if h.get("version", 1) < 2:
        print("  [!] server.py가 구버전(v1)입니다 — 장수/해상도 제한이 그대로 걸립니다.")
        print("      노트북의 server.py를 v2로 교체하세요.")
    else:
        print(f"  장수 무제한 · 해상도 최대 {h.get('max_res')} (자동 묶음 {h.get('auto_group')}장)")
    if h.get("device") != "cuda":
        print("  [!] 노트북이 CPU 모드 — 매우 느립니다.")

    if opt["res"] is None:
        opt["res"] = h.get("max_res", 518)

    print("\n[3] 전송 → 계산")
    try:
        P, C = offload(ip, paths, **opt)
        save_ply(P, C, os.path.expanduser("~/vggt/result_chain.ply"))
    except Exception as e:
        print(f"  [실패] {e}")
        sys.exit(1)
