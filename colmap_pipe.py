"""COLMAP 정밀 경로 — 사진 폴더 → 점군 → 물체 분리 → 메쉬 → 품질 리포트.

VGGT 경로(server.py)와의 역할 분담
  VGGT   : 9초. 촬영이 제대로 됐는지 보는 미리보기. 치수는 못 믿는다.
  COLMAP : 19분. 형상·치수·STL. 이쪽이 실측 기준이다.

왜 나눴는지 (2026-08-16 실측, 같은 사진 32장)
    평평한 마운트 플레이트의 평면 이탈 RMS
      VGGT   5.13%   중심→가장자리로 단조 하강(접시 왜곡)
      COLMAP 1.37%   그런 패턴 없음
  원인을 하나씩 배제했다. 포즈를 COLMAP 것으로 바꿔도(5.95%), 프레임별
  스케일까지 맞춰도(3.99%) 접시 패턴이 남았다. 즉 VGGT의 깊이맵 한 장 한 장에
  방사형 편향이 이미 들어 있어서, 뒤에서 뭘 해도 못 편다.
  (프레임 내 형상 일치는 2.0%로 괜찮다 — 그래서 미리보기로는 쓸 만하다.)

사용법
    python colmap_pipe.py <사진폴더> [--sam "blue gear|the silver metal object"]
    결과는 <사진폴더>/_out/ 에 쓴다. 입력 폴더는 건드리지 않는다.
"""
import os, sys, re, json, time, shutil, subprocess, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# COLMAP 실행 파일을 찾는 순서. 환경변수 > 알려진 위치 > PATH
COLMAP_CANDIDATES = [
    os.environ.get("COLMAP_EXE", ""),
    os.path.join(os.path.expanduser("~"), "Desktop", "스캐너_전달", "_tools", "bin", "colmap.exe"),
    r"C:\Program Files\COLMAP\bin\colmap.exe",
    "colmap",
]


def find_colmap():
    for c in COLMAP_CANDIDATES:
        if not c:
            continue
        if c == "colmap":
            if shutil.which("colmap"):
                return "colmap"
        elif os.path.exists(c):
            return c
    raise RuntimeError(
        "colmap.exe 를 못 찾았습니다. 환경변수 COLMAP_EXE 에 경로를 넣거나\n"
        "  https://github.com/colmap/colmap/releases 에서 windows-cuda 판을 받으세요.")


def run(cmd, log, tail=6):
    """COLMAP 한 단계 실행. 실패하면 마지막 줄들을 보여주고 예외."""
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        for line in out.strip().split("\n")[-12:]:
            log("    | " + line)
        raise RuntimeError(f"COLMAP 실패: {os.path.basename(cmd[1] if len(cmd)>1 else cmd[0])}")
    return out


# ---------------- 입력 정리 ----------------
def collect_images(folder, work, log=print):
    """사진만 골라 작업 폴더로 복사한다.

    ⚠ 재귀로 읽으면 안 된다. 결과물(_out/preview.png 등)이 사진으로 딸려 들어가면
    비율이 달라 COLMAP 의 single_camera 검사가 깨지고, 등록 이미지가 0이 된다.
    (2026-08-16 실제로 여기서 한 번 실패했다.)
    """
    from PIL import Image
    names = sorted(n for n in os.listdir(folder)
                   if n.lower().endswith((".jpg", ".jpeg"))
                   and os.path.isfile(os.path.join(folder, n)))
    if len(names) < 5:
        raise RuntimeError(f"사진이 {len(names)}장뿐입니다 (jpg만 셉니다). 최소 5장 필요")

    ars = []
    for n in names:
        with Image.open(os.path.join(folder, n)) as im:
            ars.append(im.size[0] / im.size[1])
    if max(ars) / min(ars) > 1.05:
        common = max(set(round(a, 2) for a in ars),
                     key=lambda v: sum(1 for a in ars if round(a, 2) == v))
        odd = [names[i] for i, a in enumerate(ars) if round(a, 2) != common]
        raise RuntimeError("가로세로 비율이 다른 파일이 섞였습니다: "
                           + ", ".join(odd[:5]) + " → 폴더에서 빼주세요")

    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir, exist_ok=True)
    for n in names:
        shutil.copy2(os.path.join(folder, n), os.path.join(img_dir, n))
    log(f"  입력 {len(names)}장 (비율 {ars[0]:.3f})")
    return img_dir, names


# ---------------- COLMAP ----------------
def sfm(cm, work, img_dir, log=print):
    db = os.path.join(work, "db.db")
    sp = os.path.join(work, "sparse")
    os.makedirs(sp, exist_ok=True)
    t0 = time.time()
    run([cm, "feature_extractor", "--database_path", db, "--image_path", img_dir,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "SIMPLE_RADIAL"], log)
    run([cm, "exhaustive_matcher", "--database_path", db], log)
    out = run([cm, "mapper", "--database_path", db, "--image_path", img_dir,
               "--output_path", sp], log)
    models = [d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp, d))]
    if not models:
        raise RuntimeError("SfM 실패 — 카메라 위치를 못 찾았습니다. "
                           "사진이 너무 흔들렸거나 겹침이 부족합니다")

    # ⚠ 반드시 '가장 큰' 모델을 골라야 한다. 이름순 첫 번째가 아니다.
    # COLMAP은 이미지들이 하나로 안 이어지면 모델을 여러 개로 쪼개는데,
    # 실측(0831_2044, 133장): 모델 2개가 나왔고 sparse/0 은 2장짜리 조각이라
    # 그걸 집어서 점 264개짜리 결과가 나왔다. 촬영 각도가 넓을수록(R 30~150°)
    # 극단 링끼리 연결이 끊겨 쪼개질 확률이 높다.
    sizes = []
    for d in models:
        n = 0
        try:
            info = run([cm, "model_analyzer", "--path", os.path.join(sp, d)], log)
            # ⚠ split(":")[1] 로 자르면 안 된다. COLMAP 로그 줄은
            #   "I20260901 22:21:34.069747 29344 model.cc:446] Registered images: 81"
            #   처럼 콜론이 여러 개라 타임스탬프의 '분'을 읽게 된다.
            #   실제로 그 버그로 모델 크기가 3/4/5/13/27장으로 보고됐는데
            #   전부 시각의 분 값이었고, 모델 선택이 무작위가 됐다.
            m_ = re.search(r"Registered images:\s*(\d+)", info)
            if m_:
                n = int(m_.group(1))
        except Exception:
            pass
        sizes.append((n, d))
    sizes.sort(reverse=True)
    best_n, best = sizes[0]
    log(f"  SfM {time.time()-t0:.0f}s · 모델 {len(models)}개 "
        f"{[f'{d}:{n}장' for n, d in sizes]} → {best} 사용")
    if len(models) > 1:
        lost = sum(n for n, _ in sizes[1:])
        log(f"  [!] 사진이 하나로 안 이어져 {lost}장이 다른 조각에 남았습니다 "
            f"(겹침 부족 또는 각도 급변)")
    if best_n < 5:
        raise RuntimeError(f"가장 큰 모델도 {best_n}장뿐입니다 — 정합 실패")
    return os.path.join(sp, best)


def dense(cm, work, img_dir, model, max_size=1600, log=print):
    d = os.path.join(work, "dense")
    t0 = time.time()
    run([cm, "image_undistorter", "--image_path", img_dir, "--input_path", model,
         "--output_path", d, "--output_type", "COLMAP",
         "--max_image_size", str(max_size)], log)
    log(f"  undistort 완료 · patch_match 시작 (오래 걸립니다)")
    run([cm, "patch_match_stereo", "--workspace_path", d,
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true"], log)
    fused = os.path.join(d, "fused.ply")
    run([cm, "stereo_fusion", "--workspace_path", d, "--workspace_format", "COLMAP",
         "--input_type", "geometric", "--output_path", fused], log)
    txt = os.path.join(d, "sparse_txt")
    os.makedirs(txt, exist_ok=True)
    run([cm, "model_converter", "--input_path", os.path.join(d, "sparse"),
         "--output_path", txt, "--output_type", "TXT"], log)
    log(f"  Dense {time.time()-t0:.0f}s → {fused}")
    return d, fused, txt


# ---------------- 카메라 모델 ----------------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]])


def read_model(txt_dir):
    """undistort 후 모델은 PINHOLE 이다. (뷰 목록, 카메라) 반환"""
    cams = {}
    for line in open(os.path.join(txt_dir, "cameras.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        p = [float(x) for x in t[4:]]
        cams[int(t[0])] = dict(w=int(t[2]), h=int(t[3]),
                               fx=p[0], fy=p[1], cx=p[2], cy=p[3])
    views = []
    lines = [l for l in open(os.path.join(txt_dir, "images.txt"), encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):          # 홀수 줄은 2D 점 목록이라 건너뜀
        t = lines[i].split()
        views.append(dict(R=qvec2rotmat([float(x) for x in t[1:5]]),
                          t=np.array([float(x) for x in t[5:8]]),
                          cam=cams[int(t[8])], name=t[9]))
    return views, cams


# ---------------- 물체 분리 ----------------
def make_masks(dense_dir, prompt, log=print):
    """undistorted 이미지에 SAM 3 실행 → {파일명: bool 마스크}"""
    import glob
    sys.path.insert(0, HERE)
    import sam_mask
    ok, why = sam_mask.available()
    if not ok:
        log(f"  [SAM 없음] {why} → 마스크 없이 진행")
        return None
    imgs = sorted(glob.glob(os.path.join(dense_dir, "images", "*.jpg")))
    try:
        masks = sam_mask.segment(imgs, prompt=prompt, device="cuda", log=log)
    except Exception as e:
        log(f"  [SAM 실패] {e} → 마스크 없이 진행")
        return None
    finally:
        try:
            sam_mask.unload()
        except Exception:
            pass
    return {os.path.basename(p): m for p, m in zip(imgs, masks)}


def filter_by_masks(fused_ply, views, masks, min_frac=0.6, min_views=3, log=print):
    """마스크 + 카메라 포즈로 물체만 남긴다.

    각 3D 점을 모든 뷰에 투영해서 '보이는 뷰 중 마스크 안에 든 비율'을 센다.
    평면 맞추고 반경으로 자르는 방식보다 훨씬 정확하다 — 물체가 판 중앙에
    없어도, 형태가 복잡해도 그대로 동작한다.
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(fused_ply)     # 법선까지 읽는다
    P = np.asarray(pcd.points)
    if masks is None:
        return pcd
    N = np.asarray(pcd.normals)
    C = np.asarray(pcd.colors)

    # 확장자를 뗀 이름으로도 찾는다.
    #
    # 왜 — gof_prep.bake_alpha 가 마스크를 알파로 구워 넣으면서 이미지를
    # jpg → png 로 바꾸고 images.txt 의 NAME 도 같이 고친다. 그런데 마스크
    # 캐시(sam_masks.npz)의 키는 jpg 인 채로 남는다. 그러면 여기 masks.get()
    # 이 전부 None 을 돌려주고, inside 가 0 이 되고, 아래 폴백이 조용히
    # '점군 전체'를 반환한다. 실측(0831_2058, 2026-09-03):
    #   마스크 키 116개(.jpg) vs 뷰 이름 115개(.png) → 교집합 0
    #   → 배경 포함 123만 점이 그대로 GOF 학습에 들어갔다.
    stem = {}
    for k, mm in masks.items():
        stem.setdefault(os.path.splitext(k)[0], mm)

    n_hit = 0
    inside = np.zeros(len(P), np.int32)
    visible = np.zeros(len(P), np.int32)
    for v in views:
        m = masks.get(v["name"])
        if m is None:
            m = stem.get(os.path.splitext(v["name"])[0])
        if m is None:
            continue
        n_hit += 1
        c = v["cam"]
        X = (v["R"] @ P.T).T + v["t"]
        z = X[:, 2]
        front = z > 1e-6
        u = np.full(len(P), -1.0); vv = np.full(len(P), -1.0)
        u[front] = c["fx"] * X[front, 0] / z[front] + c["cx"]
        vv[front] = c["fy"] * X[front, 1] / z[front] + c["cy"]
        inb = front & (u >= 0) & (u < c["w"]) & (vv >= 0) & (vv < c["h"])
        visible += inb
        idx = np.where(inb)[0]
        if len(idx):
            inside[idx] += m[vv[idx].astype(int), u[idx].astype(int)].astype(np.int32)
    if n_hit == 0:
        raise RuntimeError(
            f"마스크와 뷰 이름이 하나도 안 맞습니다 "
            f"(마스크 {len(masks)}개, 뷰 {len(views)}개). "
            f"예: 마스크 {list(masks)[:2]} vs 뷰 {[v['name'] for v in views[:2]]}")
    if n_hit < len(views):
        log(f"  [!] {len(views)}뷰 중 {n_hit}뷰에만 마스크가 있습니다")

    frac = np.where(visible > 0, inside / np.maximum(visible, 1), 0.0)
    sel = (frac >= min_frac) & (inside >= min_views)
    if sel.sum() < 500:
        # 여기서 '전체를 쓴다'는 건 배경까지 전부 통과시킨다는 뜻이다. 예전에는
        # 이 줄이 [!] 한 줄로 지나가서, 방 전체가 GOF 학습에 들어간 걸 아무도
        # 몰랐다. 마스크가 물체를 못 잡았다는 신호이므로 크게 알린다.
        log(f"  [!!] 마스크로 거르면 {int(sel.sum())}점뿐입니다 "
            f"(frac 최대 {frac.max():.2f}, 기준 {min_frac}).")
        log( "       마스크가 물체가 아닌 다른 것을 잡았을 가능성이 높습니다.")
        log( "       배경을 포함한 점군 전체를 그대로 돌려줍니다 — 결과를 믿지 마세요.")
        return pcd
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P[sel])
    out.normals = o3d.utility.Vector3dVector(N[sel])
    out.colors = o3d.utility.Vector3dVector(C[sel])
    out, _ = out.remove_statistical_outlier(30, 2.0)
    frac_kept = len(out.points) / max(len(P), 1)
    log(f"  물체 분리 {len(P):,} → {len(out.points):,}점 ({frac_kept*100:.1f}%)")
    # 마스크가 엉뚱한 걸 잡았는지 자동 점검. 실측 기준:
    #   0831_2044 (정상)  10.0% 남김 → 형상 제대로 나옴
    #   0831_2058 (실패)   0.9% 남김 → 프롬프트가 리그 부품을 잡아 결과가 쓰레기
    # 사람이 로그를 안 봐도 report.json 에 남도록 경고를 띄운다.
    if frac_kept < 0.02:
        log(f"  [!!] 남은 점이 {frac_kept*100:.1f}% 뿐입니다 — SAM 프롬프트가 물체가 아닌")
        log( "       다른 것을 잡았을 가능성이 높습니다. 결과를 믿지 마세요.")
        log( "       프롬프트를 바꿔 다시 돌리세요 (예: 'the object in the center')")
    return out


# ---------------- 메쉬 ----------------
def build_mesh(pcd, depth=9, trim=0.02, smooth=10, log=print):
    """COLMAP 법선을 '그대로' 써서 Poisson.

    ⚠ estimate_normals 로 다시 뽑으면 안 된다. COLMAP MVS 는 여러 시점에서
    일관된 법선을 주는데, 이웃점만 보는 추정은 실측 24.7% 가 30° 이상 어긋났다.
    Poisson 품질은 법선이 거의 전부라 이 차이가 그대로 표면에 나온다.
    """
    import open3d as o3d
    if not pcd.has_normals():
        log("  [!] 법선이 없어 추정합니다 (품질 저하)")
        diag = float(np.linalg.norm(np.asarray(pcd.points).max(0)
                                    - np.asarray(pcd.points).min(0)))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=diag * 0.025, max_nn=30))
    t0 = time.time()
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    m.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    bb = pcd.get_axis_aligned_bounding_box()
    m = m.crop(bb.scale(1.03, bb.get_center()))
    m.remove_degenerate_triangles(); m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    lab = np.asarray(m.cluster_connected_triangles()[0])
    if len(lab):
        m.remove_triangles_by_mask(lab != np.bincount(lab).argmax())
        m.remove_unreferenced_vertices()
    if smooth > 0:
        m = m.filter_smooth_taubin(number_of_iterations=int(smooth))
        m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()

    # 정점 색 — 회색 무광으로 보면 잡티가 다 드러난다. Meshroom 도 마지막이 텍스처링이다.
    tree = o3d.geometry.KDTreeFlann(pcd)
    src_c = np.asarray(pcd.colors)
    V = np.asarray(m.vertices)
    col = np.full((len(V), 3), 0.6)
    if len(src_c) == len(pcd.points):
        for i, p in enumerate(V):
            k, idx, _ = tree.search_knn_vector_3d(p, 6)
            if k:
                col[i] = src_c[np.asarray(idx)].mean(0)
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    m.compute_vertex_normals()
    log(f"  메쉬 {time.time()-t0:.0f}s · 삼각형 {len(m.triangles):,}")
    return m


# ---------------- 품질 ----------------
def quality(pcd, log=print):
    """평평해야 할 면이 실제로 평평한지. 눈대중 대신 이 숫자로 판단한다.

    은색(무채색·밝음) 점을 마운트 플레이트로 보고 평면을 맞춘다.
    '접시패턴'은 중심에서 가장자리로 갈수록 한 방향으로 단조롭게 휘는 것으로,
    잡음이 아니라 체계적 왜곡이라 후처리로 못 편다.
    """
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors) * 255.0
    rep = {"n_points": int(len(P))}
    if len(C) != len(P):
        return rep
    mx, mn = C.max(1), C.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    sel = (sat < 0.18) & (mx > 90)
    if sel.sum() < 500:
        log("  [품질] 평면 기준면(은색)을 못 찾아 측정 생략")
        return rep
    S = P[sel]
    c = S.mean(0); Q = S - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[2]; d = Q @ n
    span = float(np.linalg.norm(S.max(0) - S.min(0)))
    rms = float(np.sqrt((d ** 2).mean()) / span * 100)
    r = np.linalg.norm(Q - np.outer(d, n), axis=1)
    b = np.percentile(r, [0, 20, 40, 60, 80, 100])
    devs = []
    for i in range(5):
        s2 = (r >= b[i]) & (r < b[i + 1])
        devs.append(float(d[s2].mean()) if s2.sum() > 10 else float("nan"))
    mono = (all(devs[i] >= devs[i + 1] for i in range(4))
            or all(devs[i] <= devs[i + 1] for i in range(4)))
    rep.update(plane_points=int(sel.sum()), plane_rms_pct=round(rms, 2),
               dish=bool(mono), profile=[round(v, 5) for v in devs])
    log(f"  [품질] 평면 잔차 RMS {rms:.2f}% · 접시패턴 {'있음' if mono else '없음'}")
    if rms > 3.0:
        log("         3% 넘습니다 — 치수로 쓰기엔 위험합니다")
    return rep


# ---------------- 전체 ----------------
def run_all(folder, sam_prompt="", out_dir=None, max_size=1600,
            depth=9, smooth=10, keep_work=False, log=print):
    folder = os.path.abspath(os.path.expanduser(folder))
    out_dir = out_dir or os.path.join(folder, "_out")
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "colmap")
    if os.path.exists(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    cm = find_colmap()
    t_all = time.time()
    log(f"[COLMAP] {cm}")
    img_dir, names = collect_images(folder, work, log)
    model = sfm(cm, work, img_dir, log)
    dense_dir, fused, txt = dense(cm, work, img_dir, model, max_size, log)
    views, _ = read_model(txt)

    masks = make_masks(dense_dir, sam_prompt, log) if sam_prompt else None
    obj = filter_by_masks(fused, views, masks, log=log)

    import open3d as o3d
    p_obj = os.path.join(out_dir, "object.ply")
    o3d.io.write_point_cloud(p_obj, obj)
    mesh = build_mesh(obj, depth=depth, smooth=smooth, log=log)
    p_mesh = os.path.join(out_dir, "mesh.ply")
    p_stl = os.path.join(out_dir, "mesh.stl")
    o3d.io.write_triangle_mesh(p_mesh, mesh)
    o3d.io.write_triangle_mesh(p_stl, mesh)

    rep = quality(obj, log)
    try:
        import open3d as _o3
        n_dense = len(_o3.io.read_point_cloud(fused).points)
        rep["dense_points"] = int(n_dense)
        rep["mask_kept_pct"] = round(len(obj.points) / max(n_dense, 1) * 100, 2)
        rep["mask_suspect"] = bool(rep["mask_kept_pct"] < 2.0)
    except Exception:
        pass
    rep.update(images=len(names), views=len(views), elapsed_sec=round(time.time() - t_all, 1),
               n_triangles=int(len(mesh.triangles)), sam_prompt=sam_prompt,
               watertight=bool(mesh.is_watertight()))
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)     # dense 는 수 GB라 기본은 지운다
    log(f"[완료] {rep['elapsed_sec']:.0f}s → {out_dir}")
    return dict(object_ply=p_obj, mesh_ply=p_mesh, mesh_stl=p_stl, report=rep)


def main():
    ap = argparse.ArgumentParser(description="COLMAP 정밀 복원")
    ap.add_argument("folder")
    ap.add_argument("--sam", default="", help='SAM 프롬프트. 여러 개는 | 로 구분')
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--keep-work", action="store_true", help="dense 중간산물 보존")
    a = ap.parse_args()
    try:
        run_all(a.folder, a.sam, max_size=a.max_size, depth=a.depth,
                smooth=a.smooth, keep_work=a.keep_work)
    except Exception as e:
        print(f"\n[실패] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
