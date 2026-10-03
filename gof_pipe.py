"""GOF 데이터셋 → 학습 → 메쉬 추출 → 정리 → STL.

    python gof_prep.py <사진폴더>            # 먼저 데이터셋을 만들고
    python gof_pipe.py <사진폴더>/_gof       # 여기서 STL 까지

왜 두 단계로 나눴나
  학습이 30~60분 걸린다. 데이터셋 조립(1~2분)에서 좌표계가 어긋났으면 그걸
  학습 다 끝나고 알게 되는 건 최악이다. prep 이 bbox 점검을 먼저 통과시킨다.

Poisson 과 뭐가 다른가
  Poisson 은 점의 위치와 법선만 보고 껍질을 씌운다 — 점군이 부풀어 있으면
  부푼 채로 매끈하게 씌운다. GOF 는 사진을 다시 렌더해보면서 불투명도장을
  맞추고, 그 등위면을 사면체 분할로 뽑는다. 그래서 사진에 안 맞는 부풀음은
  줄어들지만, **스케일은 여전히 입력 포즈에서 온다**. VGGT 포즈로 돌리면
  VGGT 의 스케일 오차를 그대로 물려받는다 — 치수는 --size-mm 로 앵커를 걸어라.
"""
import os, sys, re, json, time, glob, subprocess, argparse
import numpy as np

# 콘솔이 cp949 면 '—' 같은 문자에서 죽는다. .bat 는 chcp 65001 을 걸어주지만
# 직접 실행할 때를 대비해 여기서도 한 번 더 막는다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


HERE = os.path.dirname(os.path.abspath(__file__))

GOF_CANDIDATES = [
    os.environ.get("GOF_DIR", ""),
    os.path.join(os.path.expanduser("~"), "gaussian-opacity-fields"),
    os.path.join(os.path.expanduser("~"), "Desktop", "gaussian-opacity-fields"),
    os.path.join(HERE, "gaussian-opacity-fields"),
]


def find_gof():
    for c in GOF_CANDIDATES:
        if c and os.path.exists(os.path.join(c, "train.py")):
            return c
    raise RuntimeError(
        "GOF 를 못 찾았습니다. 먼저 설치하세요:\n"
        "    python gof_setup.py\n"
        "이미 있다면 환경변수 GOF_DIR 에 경로를 넣으세요.")


def gof_python():
    """GOF 확장(CUDA)이 설치된 파이썬. 없으면 지금 인터프리터."""
    p = os.environ.get("GOF_PYTHON", "")
    return p if p and os.path.exists(p) else sys.executable


def run(cmd, cwd, log, tail=15, progress_every=60):
    """긴 학습이라 출력을 흘려보내면서 돌린다.

    ⚠ 줄 단위(`for line in p.stdout`)로 읽으면 안 된다.
    train.py 의 진행 표시는 tqdm 이고, tqdm 은 매 갱신을 '\\r' 로만 끝낸다.
    '\\n' 은 거의 안 나오므로 줄 단위 반복은 30000 iter 가 끝날 때까지 아무것도
    돌려주지 않는다. 실제로 그래서 학습 한 시간 동안 진행률을 못 봤다.
    여기서는 덩어리로 읽어 '\\r' 과 '\\n' 둘 다에서 끊는다.
    """
    log("  > " + " ".join(os.path.basename(c) if i == 1 else c
                          for i, c in enumerate(cmd)))
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=0)
    buf, pending, last = [], "", 0.0
    t0 = time.time()
    while True:
        chunk = p.stdout.read(256)
        if not chunk:
            break
        pending += chunk
        parts = re.split(r"[\r\n]", pending)
        pending = parts.pop()                 # 마지막 조각은 아직 안 끝난 줄
        for line in parts:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            if len(buf) > 400:
                buf.pop(0)
            now = time.time()
            if re.search(r"error|Error|Traceback|out of memory", line):
                log("    | " + line[:160])
            elif re.search(r"\[ITER|Saving|Training progress|it/s|%\|", line) \
                    and now - last >= progress_every:
                last = now
                log(f"    | [{(now-t0)/60:.0f}분] " + line[:120])
    p.wait()
    if p.returncode != 0:
        for l in buf[-tail:]:
            log("    | " + l[:200])
        raise RuntimeError(f"GOF 실패 (exit {p.returncode}): {os.path.basename(cmd[1])}")
    return "\n".join(buf)


# ---------------- 메쉬 정리 ----------------
def clean_mesh(path_in, init_ply=None, smooth=0, keep_largest=True, log=print):
    """GOF 사면체 추출 결과를 STL 로 쓸 만하게 다듬는다.

    사면체 분할은 카메라 절두체 전체를 덮으므로 물체 주변 허공에 얇은 껍질이
    같이 딸려 나온다. 초기 점군 bbox 로 자르고 가장 큰 덩어리만 남긴다.
    """
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path_in)
    n0 = len(m.triangles)
    if n0 == 0:
        raise RuntimeError(f"메쉬가 비어 있습니다: {path_in}")

    if init_ply and os.path.exists(init_ply):
        pc = o3d.io.read_point_cloud(init_ply)
        bb = pc.get_axis_aligned_bounding_box()
        bb = bb.scale(1.05, bb.get_center())     # 5% 여유 — 표면이 점군보다 조금 밖이다
        m = m.crop(bb)

    m.remove_degenerate_triangles()
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_unreferenced_vertices()

    if keep_largest and len(m.triangles):
        lab, cnt, _ = m.cluster_connected_triangles()
        lab = np.asarray(lab)
        if len(cnt) > 1:
            m.remove_triangles_by_mask(lab != int(np.argmax(cnt)))
            m.remove_unreferenced_vertices()

    if smooth > 0:
        m = m.filter_smooth_taubin(number_of_iterations=int(smooth))
        m.remove_degenerate_triangles()
        m.remove_unreferenced_vertices()

    m.compute_vertex_normals()
    log(f"  정리 삼각형 {n0:,} → {len(m.triangles):,} · "
        f"watertight {'예' if m.is_watertight() else '아니오'}")
    return m


def apply_scale(m, scale=None, size_mm=None, log=print):
    """치수 앵커. GOF 좌표는 임의 스케일이라 뭐라도 걸어야 mm 가 된다."""
    if size_mm:
        ext = m.get_axis_aligned_bounding_box().get_extent()
        cur = float(max(ext))
        if cur <= 0:
            raise RuntimeError("메쉬 크기가 0 입니다")
        s = size_mm / cur
        log(f"  스케일 앵커: 최대변 {cur:.5f} → {size_mm}mm (x{s:.3f})")
    elif scale:
        s = float(scale)
        log(f"  스케일 x{s}")
    else:
        log("  [!] 스케일 앵커 없음 — STL 은 임의 단위입니다 (--size-mm 을 쓰세요)")
        return m
    m.scale(s, center=(0, 0, 0))
    return m


def scene_voxel(ds, div=512, pct=5.0, log=print):
    """초기 점군 크기에서 TSDF 복셀 크기를 정한다.

    ⚠ min/max 전체 범위를 쓰면 안 된다.
    COLMAP sparse 점군에는 배경·오정합 점이 멀리 흩어져 있어서 bbox 가
    물체보다 훨씬 커진다. 실측(0831_2058):

        전체 min/max   122.26     <- 이걸 쓰면 복셀 0.2388
        5~95 퍼센타일    9.25
        실제 물체        7.50     <- 물체를 가로지르는 복셀이 31개뿐

    그래서 가우시안을 4배 더 학습시켜도(686s -> 1764s) 삼각형이 2,907 -> 3,053
    으로 5% 밖에 안 늘었다. 메쉬 해상도를 정하는 건 가우시안 수가 아니라
    이 복셀 크기였다. 퍼센타일로 이상치를 떼고 잡는다.
    """
    import open3d as o3d
    p = os.path.join(ds, "sparse", "0", "points3D.ply")
    P = np.asarray(o3d.io.read_point_cloud(p).points)
    full = float((P.max(0) - P.min(0)).max())
    span = float((np.percentile(P, 100 - pct, axis=0)
                  - np.percentile(P, pct, axis=0)).max())
    if span <= 0:
        span = full
    v = span / div
    log(f"  복셀 {v:.6g} ({pct:g}~{100-pct:g}% 범위 {span:.4g} / {div} · "
        f"전체범위는 {full:.4g})")
    return v


# ---------------- 전체 ----------------
def run_all(ds, out_dir=None, iters=30000, res=2, extract_iter=None,
            smooth=0, scale=None, size_mm=None, skip_train=False,
            extract="auto", voxel=None, train_extra=None, log=print):
    ds = os.path.abspath(os.path.expanduser(ds))
    if not os.path.isdir(os.path.join(ds, "sparse", "0")):
        raise RuntimeError(f"GOF 데이터셋이 아닙니다 (sparse/0 없음): {ds}\n"
                           f"  먼저: python gof_prep.py <사진폴더>")
    out_dir = out_dir or os.path.join(ds, "_model")
    os.makedirs(out_dir, exist_ok=True)

    # VGGT 경로는 이미지가 이미 518x294 다. 거기에 -r 2 를 걸면 259x147 이 되어
    # 메쉬 디테일이 남지 않는다. -r 은 '큰 사진을 8GB 에 맞추려고' 있는 값이다.
    try:
        for line in open(os.path.join(ds, "sparse", "0", "cameras.txt"), encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            w = int(line.split()[2])
            if w <= 800 and res > 1:
                log(f"  [!] 이미지 가로가 {w}px 인데 -r {res} 면 {w//res}px 까지 줄어듭니다.")
                log(f"      VGGT 경로처럼 이미 작은 입력이면 -r 1 을 쓰세요.")
            break
    except Exception:
        pass

    gof, py = find_gof(), gof_python()
    extract_iter = extract_iter or iters
    log(f"[GOF] {gof}")
    log(f"[python] {py}")

    t_all = time.time()
    if not skip_train:
        log(f"\n[1/3] 학습 {iters} iter (수십 분 걸립니다)")
        t0 = time.time()
        # ⚠ --save_iterations 를 반드시 같이 넘긴다.
        # train.py 의 기본값은 [7000, 30000] 이라, --iterations 만 바꾸면
        # 그 지점에 체크포인트가 안 남는다. extract_mesh.py 는
        # point_cloud/iteration_{N}/point_cloud.ply 를 직접 여니까 곧바로 실패한다.
        tcmd = [py, os.path.join(gof, "train.py"), "-s", ds, "-m", out_dir,
                "-r", str(res), "--iterations", str(iters),
                "--save_iterations", str(iters),
                "--test_iterations", "-1"]
        tcmd += list(train_extra or [])
        run(tcmd, gof, log)
        log(f"  학습 {time.time()-t0:.0f}s")
    else:
        log("\n[1/3] 학습 건너뜀 (--skip-train)")

    # 추출기 선택. GOF 는 두 가지를 준다.
    #   extract_mesh.py       사면체 분할. 품질은 이쪽이 낫지만
    #                         tetranerf 확장(CGAL+cnpy)이 필요하다 — 윈도우에서 잘 막힌다.
    #   extract_mesh_tsdf.py  TSDF 융합. torch + open3d 만 쓴다. 확장 불필요.
    # auto 는 tetra 확장이 실제로 import 되는지 보고 고른다.
    mode = extract
    if mode == "auto":
        rc = subprocess.run([py, "-c", "from tetranerf.utils.extension import cpp"],
                            capture_output=True, text=True).returncode
        mode = "tetra" if rc == 0 else "tsdf"
        if mode == "tsdf":
            log("  [알림] tetranerf 확장이 없어 TSDF 추출로 갑니다 "
                "(사면체 분할을 쓰려면 tetra-triangulation 을 빌드하세요)")
    script = "extract_mesh.py" if mode == "tetra" else "extract_mesh_tsdf.py"

    log(f"\n[2/3] 메쉬 추출 ({'사면체 분할' if mode == 'tetra' else 'TSDF 융합'})")
    t0 = time.time()
    # ⚠ '새로 생긴 ply' 만 찾으면 안 된다. 재추출하면 tsdf.ply 를 덮어쓰므로
    # 파일 목록이 그대로라 후보가 0개가 되어버린다(--skip-train 에서 실제로 겪음).
    # 새로 생긴 것 + 이번 실행 중에 수정된 것을 함께 본다.
    before = {p: os.path.getmtime(p) for p in
              glob.glob(os.path.join(out_dir, "**", "*.ply"), recursive=True)}
    t_start = time.time()
    cmd = [py, os.path.join(gof, script), "-m", out_dir,
           "--iteration", str(extract_iter)]

    if mode == "tsdf":
        # GOF 의 voxel_size 기본값 0.002 는 DTU 처럼 정규화된 장면 기준이다.
        # 여기 COLMAP 장면은 크기가 165 단위라 그대로 두면 축마다 8만 복셀을
        # 요구해서 block_count 를 넘겨 죽는다. 초기 점군 크기에서 유도한다.
        v = voxel or scene_voxel(ds, log=log)
        cmd += ["--voxel_size", f"{v:.6g}"]

    run(cmd, gof, log)
    # 출력 경로가 GOF 버전마다 달라서 하드코딩하지 않고 이번에 쓰인 ply 를 찾는다
    touched = []
    for p in glob.glob(os.path.join(out_dir, "**", "*.ply"), recursive=True):
        if "point_cloud" in p.replace("\\", "/"):
            continue                      # 가우시안 체크포인트지 메쉬가 아니다
        if p not in before or os.path.getmtime(p) > t_start - 1:
            touched.append(p)
    cand = touched
    if not cand:
        raise RuntimeError(f"추출된 메쉬를 못 찾았습니다. {out_dir} 안을 확인하세요.")
    raw = max(cand, key=os.path.getmtime)
    log(f"  추출 {time.time()-t0:.0f}s → {os.path.basename(raw)}")

    log("\n[3/3] 정리 + STL")
    init_ply = os.path.join(ds, "sparse", "0", "points3D.ply")
    m = clean_mesh(raw, init_ply=init_ply, smooth=smooth, log=log)
    m = apply_scale(m, scale=scale, size_mm=size_mm, log=log)

    import open3d as o3d
    p_mesh = os.path.join(ds, "mesh_gof.ply")
    p_stl = os.path.join(ds, "mesh_gof.stl")
    o3d.io.write_triangle_mesh(p_mesh, m)
    o3d.io.write_triangle_mesh(p_stl, m)

    ext = m.get_axis_aligned_bounding_box().get_extent()
    rep = dict(dataset=ds, raw_mesh=raw, extract=mode, voxel=voxel,
               train_extra=list(train_extra or []), iterations=iters, resolution_div=res,
               n_triangles=int(len(m.triangles)), n_vertices=int(len(m.vertices)),
               watertight=bool(m.is_watertight()),
               bbox=[round(float(v), 4) for v in ext],
               size_mm=size_mm, scale=scale,
               elapsed_sec=round(time.time() - t_all, 1))
    with open(os.path.join(ds, "report_gof.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return dict(mesh_ply=p_mesh, mesh_stl=p_stl, report=rep)


def main():
    ap = argparse.ArgumentParser(description="GOF 데이터셋 → 메쉬 → STL")
    ap.add_argument("dataset", help="gof_prep.py 가 만든 _gof 폴더")
    ap.add_argument("-m", "--model", default="", help="학습 출력 폴더")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--extract-iter", type=int, default=0)
    ap.add_argument("-r", "--res", type=int, default=2,
                    help="학습 해상도 축소 배수. 8GB VRAM 이면 2 이상 권장")
    ap.add_argument("--smooth", type=int, default=0, help="Taubin 반복 횟수")
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--size-mm", type=float, default=0.0,
                    help="물체 최대변의 실제 길이(mm) — 치수 앵커")
    ap.add_argument("--skip-train", action="store_true", help="학습된 모델 재사용")
    ap.add_argument("--extract", choices=["auto", "tetra", "tsdf"], default="auto",
                    help="메쉬 추출 방식. auto=tetra 확장 있으면 tetra, 없으면 tsdf")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="TSDF 복셀 크기. 0=점군 크기에서 자동 산출")
    ap.add_argument("--train-extra", default="",
                    help='train.py 로 그대로 넘길 인자. '
                         '예: "--densify_grad_threshold 0.0004"')
    a = ap.parse_args()

    try:
        res = run_all(a.dataset, out_dir=a.model or None, iters=a.iters,
                      res=a.res, extract_iter=a.extract_iter or None,
                      smooth=a.smooth, scale=a.scale or None,
                      size_mm=a.size_mm or None, skip_train=a.skip_train,
                      extract=a.extract, voxel=a.voxel or None,
                      train_extra=a.train_extra.split() if a.train_extra else None)
    except Exception as e:
        print(f"\n[실패] {e}")
        return 1

    rep = res["report"]
    print("\n" + "=" * 62)
    print("  결과")
    print("=" * 62)
    print(f"  메쉬   {res['mesh_ply']}  (삼각형 {rep['n_triangles']:,})")
    print(f"  STL    {res['mesh_stl']}")
    print(f"  크기   {rep['bbox']}  · watertight {'예' if rep['watertight'] else '아니오'}")
    print(f"  소요   {rep['elapsed_sec']:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        sys.exit(1)
