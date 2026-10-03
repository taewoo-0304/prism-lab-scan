"""
노트북 단독 테스트 — 젯슨 없이 사진 폴더로 전체 파이프라인 확인

젯슨이 하는 일(HTTP 요청)을 노트북 자기 자신에게 보냅니다. 그래서 네트워크
구간만 빼고 실제와 완전히 같은 경로를 탑니다: Flask → VGGT → 정합 → 후처리
→ 점군 → (원하면) 도면.

사용법:
  VGGT-Test.bat 더블클릭  (폴더 경로를 물어봅니다)
또는:
  python localtest.py <이미지폴더> [--sam "the object"] [--mm 150] [--drawing]

옵션:
  --sam TEXT   SAM 3로 물체만 남기기 (예: "the object on the table")
  --mm N       카메라~물체 거리 mm (poses.json 이 있으면 무시됨)
  --floor N    바닥 평면 제거 두께 (기본 15, 0=끔)
  --res N      해상도 (기본 518 = 최대)
  --drawing    3면도 PDF까지 생성
  --name TEXT  도면에 적을 물체 이름
"""
import os, sys, io, json, glob, time, base64, argparse
import numpy as np

HOST = "127.0.0.1"
PORT = 5000
HERE = os.path.dirname(os.path.abspath(__file__))


def wait_server(timeout=5):
    try:
        import requests
        r = requests.get(f"http://{HOST}:{PORT}/health", timeout=timeout)
        if r.status_code == 200 and r.json().get("alive"):
            return r.json()
    except Exception:
        pass
    return None


def save_preview(P, C, out_png, n=60000):
    """점군을 네 방향에서 본 PNG. 윈도우엔 .ply 기본 연결 프로그램이 없어서,
    뷰어를 안 깔아도 결과를 바로 볼 수 있게 만든다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if len(P) < 10:
        return None
    if len(P) > n:
        idx = np.random.default_rng(0).choice(len(P), n, replace=False)
        P, C = P[idx], C[idx]
    col = np.clip(np.asarray(C, float) / 255.0, 0, 1) if C is not None and len(C) == len(P) \
        else np.full((len(P), 3), 0.6)

    fig = plt.figure(figsize=(11, 10))
    span = (P.max(0) - P.min(0)).max() or 1.0
    mid = (P.max(0) + P.min(0)) / 2
    for k, az in enumerate((45, 135, 225, 315)):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 2], P[:, 1], s=0.4, c=col, linewidths=0)
        ax.set_title(f"{az}°", fontsize=10)
        ax.view_init(elev=20, azim=az)
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_zlim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.suptitle(f"point cloud preview  ({len(P):,} pts shown)", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.05, hspace=0.10)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--sam", nargs="?", const="", default=None,
                    help="SAM으로 물체만 남김. 값 없이 주면 중앙 박스 방식")
    ap.add_argument("--mm", type=float, default=0.0)
    ap.add_argument("--floor", type=float, default=15.0)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--maxdist", type=float, default=0.0)
    # 지금까지 이 둘을 못 보내서 서버 기본값(50/25)에 묶여 있었다. SAM을 쓰면
    # 서버가 알아서 100/90으로 풀지만, 직접 주면 그 값이 우선한다.
    ap.add_argument("--dist", type=float, default=None,
                    help="가까운 쪽 몇 %%를 남길지 (기본: 서버 판단)")
    ap.add_argument("--conf", type=float, default=None,
                    help="신뢰도 상위 몇 %%를 남길지 (기본: 서버 판단)")
    ap.add_argument("--maxpoints", type=int, default=None,
                    help="점 수 상한 (0=무제한)")
    ap.add_argument("--drawing", action="store_true")
    ap.add_argument("--capacity", action="store_true",
                    help="쪼개지 않고 한 번에 되는 최대 장수를 실측")
    ap.add_argument("--name", default="테스트물체")
    a = ap.parse_args()

    print("=" * 60)
    print("  노트북 단독 테스트")
    print("=" * 60)

    folder = os.path.expanduser(a.folder.strip().strip('"'))
    # 결과물은 입력 폴더가 아니라 하위 _vggt_out/ 에 쓴다.
    # 예전에는 preview.png 를 입력 폴더에 그대로 남겼는데, 같은 폴더를 두 번째
    # 돌리면 그 png 가 '사진'으로 다시 읽혔다. 비율이 다른 이미지가 한 장 섞이면
    # load_and_preprocess_images 가 "전 프레임"을 흰색으로 패딩해서 캔버스를 키우고,
    # 그 흰 여백에서 나온 쓰레기 점이 결과를 통째로 망친다 (실측: 점 2.4배, 시간 24배).
    outdir = os.path.join(folder, "_vggt_out")
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.jpeg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        print(f"\n[!] 사진이 없습니다: {folder}")
        print("    폴더에 jpg 또는 png 를 넣어주세요. 10~30장 권장.")
        return 1
    print(f"\n사진 {len(paths)}장  ({folder})")
    if len(paths) < 3:
        print("[!] 최소 3장은 필요합니다.")
        return 1

    # 비율이 섞여 있으면 여기서 잡는다 (서버까지 가기 전에)
    try:
        from PIL import Image
        ars = []
        for p in paths:
            with Image.open(p) as im:
                ars.append((im.size[0] / im.size[1], p))
        if max(a_ for a_, _ in ars) / min(a_ for a_, _ in ars) > 1.05:
            common = max(set(round(a_, 2) for a_, _ in ars),
                         key=lambda v: sum(1 for a_, _ in ars if round(a_, 2) == v))
            odd = [p for a_, p in ars if round(a_, 2) != common]
            print(f"\n[!] 가로세로 비율이 다른 파일이 섞여 있습니다 ({len(odd)}개):")
            for p in odd[:5]:
                print(f"      {os.path.basename(p)}")
            print("    이대로 넣으면 모든 프레임이 흰색으로 패딩돼서 결과가 크게 나빠집니다.")
            print("    사진이 아닌 파일이면 폴더에서 빼고 다시 실행하세요.")
            return 1
    except ImportError:
        pass

    print("\n[1/4] 서버 확인")
    h = wait_server()
    if not h:
        print(f"  [!] {HOST}:{PORT} 에 서버가 없습니다.")
        print("      VGGT-Server.bat 을 먼저 실행해서 켜두세요.")
        return 1
    print(f"  장치 {h['device']} · GPU {h.get('gpu')} ({h.get('vram_gb')}GB) "
          f"· 자동 묶음 {h.get('auto_group')}장")
    print(f"  SAM 3: {'사용 가능' if h.get('sam') else '없음 — ' + str(h.get('sam_reason'))}")
    if a.sam is not None and not h.get("sam"):
        print("  [!] --sam 을 줬지만 서버가 SAM을 못 씁니다. 마스크 없이 진행합니다.")

    if a.capacity:
        import requests
        print("\n[측정] 프레임 수를 늘려가며 실제 한계를 찾습니다 (몇 분 걸립니다)")
        with open(paths[0], "rb") as f:
            r = requests.post(f"http://{HOST}:{PORT}/capacity",
                              files=[("images", (os.path.basename(paths[0]), f, "image/jpeg"))],
                              data={"res": str(a.res or h.get("max_res", 518))},
                              timeout=3600)
        j = r.json()
        for line in j.get("log", []):
            print(f"  | {line}")
        if r.status_code != 200:
            print(f"\n[!] 실패: {j.get('error')}")
            return 1
        print(f"\n{'=' * 60}")
        print(f"  쪼개지 않고 한 번에: 최대 {j['max_frames']}장 @ 해상도 {j['res']}")
        print(f"  (기존 추정치는 {j['estimate_was']}장이었습니다)")
        print(f"{'=' * 60}")
        print(f"\n  촬영은 {j['max_frames']}장 이하로 하시면 단일 패스로 처리됩니다.")
        return 0

    # poses.json 있으면 자동 사용
    data = {"res": a.res or h.get("max_res", 518),
            "floor": a.floor, "noise_std": a.noise,
            "ref_mm": a.mm, "ref_mode": "d", "max_dist_mm": a.maxdist}
    # None 이면 아예 안 보낸다 — 서버가 SAM 유무를 보고 스스로 정하게 둔다
    for key, val in (("dist_pct", a.dist), ("conf_pct", a.conf),
                     ("max_points", a.maxpoints)):
        if val is not None:
            data[key] = val
    pj = os.path.join(folder, "poses.json")
    if os.path.exists(pj):
        try:
            doc = json.load(open(pj, encoding="utf-8"))
            if len(doc.get("poses", [])) == len(paths):
                data["poses_json"] = json.dumps(doc)
                print(f"  poses.json 사용 ({len(doc['poses'])}개)")
            else:
                print(f"  [!] poses.json {len(doc.get('poses', []))}개 ≠ 사진 {len(paths)}장 → 무시")
        except Exception as e:
            print(f"  [!] poses.json 읽기 실패: {e}")
    if a.sam is not None:
        data["sam"] = 1
        if a.sam:
            data["sam_prompt"] = a.sam
        else:
            data["floor"] = 0

    sam_label = ("" if a.sam is None else
                 f" · SAM '{a.sam}'" if a.sam else " · SAM 중앙박스")
    print(f"\n[2/4] 계산 요청  (해상도 {data['res']}{sam_label})")
    print("  * 처음 실행은 모델 로딩 때문에 오래 걸립니다\n")
    import requests
    files = [("images", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in paths]
    t0 = time.time()
    try:
        r = requests.post(f"http://{HOST}:{PORT}/reconstruct",
                          files=files, data={k: str(v) for k, v in data.items()},
                          timeout=7200)
    finally:
        for _, (_, f, _) in files:
            f.close()

    try:
        j = r.json()
    except ValueError:
        print(f"  [!] 서버 응답을 해석할 수 없습니다: {r.text[:300]}")
        return 1
    for line in j.get("log", []):
        print(f"  | {line}")
    if r.status_code != 200:
        print(f"\n[!] 실패: {j.get('error')}")
        return 1

    z = np.load(io.BytesIO(base64.b64decode(j["npz_b64"])))
    P, C = z["points"], z["colors"]

    print(f"\n[3/4] 결과")
    print(f"  점 {len(P):,}개 · 서버 {j['elapsed_sec']}s · 전체 {time.time()-t0:.1f}s")
    print(f"  {j['n_images']}장 @ 해상도 {j['res']} · 묶음 {j.get('group')}장")
    print(f"  스케일 기준: {j.get('scale_source')}")
    if j.get("dims_mm"):
        d = j["dims_mm"]
        print(f"  치수(mm): X {d[0]:.1f} · Y {d[1]:.1f} · Z {d[2]:.1f}")
    else:
        print("  치수: 스케일 기준이 없어 단위 없음 (--mm 이나 poses.json 필요)")
    if j.get("fit_rms_mm") is not None:
        print(f"  궤적 잔차: RMS {j['fit_rms_mm']}mm / 최대 {j['fit_max_mm']}mm")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "result.ply")
    import trimesh
    trimesh.PointCloud(vertices=P, colors=C).export(out)
    print(f"  점군 저장: {out}")

    png = save_preview(P, C, os.path.join(outdir, "preview.png"))
    if png:
        print(f"  미리보기 저장: {png}")
        try:
            os.startfile(png)
        except Exception:
            pass

    print(f"\n[4/4] 도면")
    if not a.drawing:
        print("  건너뜀 (--drawing 을 주면 생성합니다)")
    else:
        try:
            sys.path.insert(0, HERE)
            import make_drawing
            pdf = os.path.join(outdir, "도면.pdf")
            make_drawing.make_drawing(out, name=a.name, up=j.get("up") or "y",
                                      out_pdf=pdf, make_stl=False)
            print(f"  도면 저장: {pdf}")
            try:
                os.startfile(pdf)
            except Exception:
                pass
        except ImportError as e:
            print(f"  [건너뜀] 도면용 패키지가 없습니다: {e}")
            print("     setup_laptop.py 를 다시 돌리면 matplotlib/scipy/scikit-image 를 깝니다")
        except Exception as e:
            print(f"  [도면 실패] {e}")

    print("\n" + "=" * 60)
    print("테스트 완료. result.ply 를 MeshLab 등으로 열어보세요.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        sys.exit(1)
