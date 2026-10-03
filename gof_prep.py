"""점군 + 사진 → GOF 입력 데이터셋 (COLMAP sparse 규약).

왜 이 단계가 따로 필요한가
  GOF(Gaussian Opacity Fields)는 Poisson 같은 '점군 → 메쉬' 변환기가 아니다.
  3D Gaussian Splatting 계열이라 **사진을 다시 렌더링해보면서** 가우시안을
  최적화한다. 그래서 최소한 이 세 가지가 한 세트로 필요하다.

      images/                사진
      sparse/0/cameras.txt   내부 파라미터 (PINHOLE 이어야 함)
      sparse/0/images.txt    외부 파라미터 (world→cam, COLMAP 규약)
      sparse/0/points3D.ply  초기 점군

  점군은 '초기화'로만 들어간다. .ply 하나만 있으면 GOF 는 못 돈다.

⚠ 점군과 포즈는 반드시 '같은 실행'에서 나와야 한다
  COLMAP 은 매 실행마다 좌표계(원점·축·스케일)를 새로 잡는다. VGGT 도 마찬가지다.
  그래서 예전에 만들어 둔 object.ply 를 새로 계산한 포즈와 짝지으면 점군이 카메라
  앞에 있지도 않게 되고, GOF 는 그걸 조용히 받아들여 쓰레기를 학습한다.
  이 스크립트가 포즈와 점군을 항상 같은 실행에서 함께 뽑는 이유다.
  (_out 에 남은 mesh/object.ply 들은 work 폴더가 지워져서 포즈가 이미 없다 —
   colmap_pipe.run_all 의 keep_work=False)

사용법
    python gof_prep.py <사진폴더>                    # COLMAP 포즈 + sparse 점군
    python gof_prep.py <사진폴더> --dense            # COLMAP 포즈 + dense 점군(느림)
    python gof_prep.py <사진폴더> --pose vggt        # VGGT 포즈 + VGGT 점군
    python gof_prep.py <사진폴더> --ply 같은실행.ply  # 점군만 갈아끼우기
    python gof_prep.py <사진폴더> --sam              # 물체만 남기기(중앙 박스)
    python gof_prep.py <사진폴더> --sam --sam-prompt "blue gear"

⚠ --sam 을 안 쓰면 초기 점군에 턴테이블·배경이 그대로 들어간다.
  gof_pipe 의 clean_mesh 는 '가장 큰 덩어리' 를 남기므로, 결과 메쉬가 물체가
  아니라 받침 원반이 된다 (실측 0831_2058: 936k 면 중 대부분이 원반이었다).
  덤으로 점군이 좁아지면 TSDF 복셀도 물체 기준으로 잘게 잡힌다.

결과는 <사진폴더>/_gof/ 에 쓴다. 입력 폴더는 건드리지 않는다.
"""
import os, sys, json, time, shutil, argparse
import numpy as np

# 콘솔이 cp949 면 '—' 같은 문자에서 죽는다. .bat 는 chcp 65001 을 걸어주지만
# 직접 실행할 때를 대비해 여기서도 한 번 더 막는다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------- COLMAP 텍스트 규약 ----------------
def rotmat2qvec(R):
    """3x3 회전행렬 → COLMAP qvec (w,x,y,z).

    trace 가 음수일 때 sqrt 안이 음수가 되는 순진한 공식은 쓰지 않는다.
    카메라가 물체를 아래로 내려다보는 링 촬영에서는 R 이 180도 가까이 도는
    프레임이 흔해서 그 분기에 실제로 걸린다.
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q = -q
    return q


def write_cameras_txt(path, cams):
    """cams: [dict(id,w,h,fx,fy,cx,cy)]  — GOF 로더는 PINHOLE/SIMPLE_PINHOLE 만 받는다."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c in cams:
            f.write(f"{c['id']} PINHOLE {c['w']} {c['h']} "
                    f"{c['fx']:.10f} {c['fy']:.10f} {c['cx']:.10f} {c['cy']:.10f}\n")


def write_images_txt(path, views):
    """views: [dict(id,q,t,cam_id,name)]

    ⚠ 이미지 한 장당 두 줄이고, 둘째 줄(2D 점 목록)을 '빈 줄'로 쓰면 안 된다.
    VGGT 는 2D 트랙을 안 주니 비워두고 싶지만, 읽는 쪽이 두 부류다.
      - GOF/3DGS: 데이터 줄 다음 줄을 무조건 한 줄 더 읽는다 → 빈 줄도 괜찮다
      - colmap_pipe.read_model: 빈 줄을 먼저 걸러낸 뒤 2줄씩 끊는다
        → 빈 줄이면 데이터 줄만 남아 **뷰가 정확히 절반으로 조용히 줄어든다**
    실측(0831_2058, VGGT 17장): 되읽으니 9뷰가 나왔다.
    그래서 더미 관측 "0 0 -1" 을 한 개 넣어 양쪽 파서 모두에서 성립하게 한다.
    (point3D_id = -1 은 '대응하는 3D 점 없음' 이라 COLMAP 규약상으로도 유효하다.
     3DGS 는 이 값을 학습에 쓰지 않는다.)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for v in views:
            q, t = v["q"], v["t"]
            f.write(f"{v['id']} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {v['cam_id']} {v['name']}\n")
            f.write("0 0 -1\n")


def store_ply(path, xyz, rgb):
    """3DGS/GOF 의 storePly 와 같은 필드 구성으로 쓴다.

    GOF 의 씬 로더는 sparse/0/points3D.ply 가 이미 있으면 .bin/.txt 를 아예 안 읽는다.
    수십만 점을 텍스트로 쓰면 로딩만 몇 분 걸리므로 여기서 .ply 를 직접 만든다.
    법선 자리는 0 으로 둔다 — 로더가 읽기만 하고 쓰지는 않는다.

    ⚠ open3d 로 쓰지 않는다. 3DGS 계열의 fetchPly 는 x,y,z / nx,ny,nz /
    red,green,blue 라는 이름을 그대로 찾는데, 저장기마다 필드 이름·타입이
    미묘하게 달라진다. plyfile 의존도 늘리지 않으려고 numpy 로 직접 쓴다.
    """
    xyz = np.ascontiguousarray(xyz, np.float32)
    rgb = np.ascontiguousarray(rgb, np.uint8)
    if len(xyz) != len(rgb):
        raise ValueError(f"점 {len(xyz)}개 ≠ 색 {len(rgb)}개")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.zeros(len(xyz), dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(arr)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def write_points3D_txt(path, xyz, rgb, cap=200_000):
    """.ply 를 못 읽는 변형 로더를 위한 예비. 너무 크면 잘라서 쓴다."""
    n = len(xyz)
    idx = np.arange(n) if n <= cap else np.random.default_rng(0).choice(n, cap, replace=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list\n# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for j, i in enumerate(idx):
            p, c = xyz[i], rgb[i]
            f.write(f"{j+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n")


def bbox_check(xyz, cam_centers, log=print):
    """점군이 카메라들 사이에 있는지 본다 — 좌표계 불일치를 여기서 잡는다."""
    lo, hi = xyz.min(0), xyz.max(0)
    ext = float(np.linalg.norm(hi - lo))
    ctr = (lo + hi) / 2
    d = np.linalg.norm(np.asarray(cam_centers) - ctr, axis=1)
    log(f"  [점검] 점군 크기 {ext:.4f} · 중심에서 카메라까지 "
        f"{d.min():.4f}~{d.max():.4f}")
    if ext <= 0 or d.min() > ext * 20:
        log("  [!] 카메라가 점군에서 지나치게 멉니다 — 좌표계가 다를 가능성이 큽니다.")
        return False
    return True


def verify_dataset(out, log=print):
    """다 쓰고 나서 '디스크에 쓴 것을 되읽어' 검증한다.

    왜 되읽는가 — 쓰기와 읽기가 어긋나는 버그는 파일을 안 열어보면 안 잡힌다.
    실제로 여기서 잡았다: images.txt 둘째 줄을 빈 줄로 쓰니 되읽을 때 뷰가
    17개 → 9개로 조용히 반토막 났다.

    점군을 각 카메라로 재투영해서 화면 안에 떨어지는 비율을 본다. 포즈든 점군이든
    좌표계가 어긋나 있으면 이 숫자가 바로 무너진다 — 학습 한 시간 태우기 전에.
    """
    import open3d as o3d
    sp = os.path.join(out, "sparse", "0")

    cams = {}
    for line in open(os.path.join(sp, "cameras.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        p = [float(x) for x in t[4:]]
        cams[int(t[0])] = dict(w=int(t[2]), h=int(t[3]),
                               fx=p[0], fy=p[1], cx=p[2], cy=p[3])

    # ⚠ 빈 줄을 거르지 않고 '데이터 줄'만 짝수 번째로 세지 않는다.
    # 홀수 줄만 파싱하되, 파싱된 뷰 수를 실제 이미지 수와 반드시 대조한다.
    lines = [l for l in open(os.path.join(sp, "images.txt"), encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    views = []
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        R = qvec2rotmat_local([float(x) for x in t[1:5]])
        views.append(dict(R=R, t=np.array([float(x) for x in t[5:8]]),
                          cam=cams[int(t[8])], name=t[9]))

    n_img = len([f for f in os.listdir(os.path.join(out, "images"))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    ok = True
    if len(views) != n_img:
        log(f"  [!] 되읽은 뷰 {len(views)}개 ≠ 이미지 {n_img}장 — images.txt 형식이 깨졌습니다")
        ok = False

    P = np.asarray(o3d.io.read_point_cloud(os.path.join(sp, "points3D.ply")).points)
    if len(P) > 40000:
        P = P[np.random.default_rng(0).choice(len(P), 40000, replace=False)]

    inside, front = [], []
    for v in views:
        c = v["cam"]
        X = (v["R"] @ P.T).T + v["t"]                 # world → cam
        z = X[:, 2]
        okz = z > 1e-6
        zz = np.where(okz, z, 1.0)
        u = c["fx"] * X[:, 0] / zz + c["cx"]
        w = c["fy"] * X[:, 1] / zz + c["cy"]
        inside.append((okz & (u >= 0) & (u < c["w"]) & (w >= 0) & (w < c["h"])).mean())
        front.append(okz.mean())
    inside, front = np.array(inside), np.array(front)

    log(f"  [검증] 뷰 {len(views)}개 · 카메라 앞 {front.mean()*100:.1f}% · "
        f"화면 안 {inside.mean()*100:.1f}% (최소 {inside.min()*100:.1f}%)")
    if front.mean() < 0.8 or inside.mean() < 0.3:
        log("  [!] 점군이 카메라에서 제대로 안 보입니다 — 포즈/점군 좌표계 불일치입니다.")
        log("      이대로 학습하면 결과가 망가집니다. 중단하세요.")
        ok = False
    return ok, dict(n_views=len(views), n_images=n_img,
                    front_pct=round(float(front.mean()) * 100, 1),
                    inside_pct=round(float(inside.mean()) * 100, 1),
                    inside_min_pct=round(float(inside.min()) * 100, 1))


def qvec2rotmat_local(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]])


def read_points3D_txt(path):
    xyz, rgb = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        xyz.append([float(t[1]), float(t[2]), float(t[3])])
        rgb.append([int(t[4]), int(t[5]), int(t[6])])
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.uint8)


def read_cam_centers_txt(path):
    """images.txt → 카메라 중심 C = -R^T t"""
    import colmap_pipe
    lines = [l for l in open(path, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    out = []
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        R = colmap_pipe.qvec2rotmat([float(x) for x in t[1:5]])
        tv = np.array([float(x) for x in t[5:8]])
        out.append(-R.T @ tv)
    return np.asarray(out)


def bake_alpha(img_dir, images_txt, masks, log=print):
    """SAM 마스크를 이미지의 알파 채널로 구워 넣고 images.txt 이름을 .png 로 고친다.

    왜 이게 필요한가 — 점군만 걸러서는 부족하다.
    초기 점군에서 배경을 빼도 GOF 는 '마스크 안 된 원본 이미지' 로 학습하므로
    배경까지 그대로 모델링한다. 그러면 extract_mesh_tsdf 가 장면 전체를 물체용
    미세 복셀로 담으려다 블록 수가 폭발해서 죽는다
    (실측: voxel 0.0026 에서 Open3D 가 9.8GB 를 할당하려다 exit 0xC0000005,
     0.008 로 굵혀도 마찬가지였다).

    GOF/3DGS 는 이미 해결책을 갖고 있다.
      - scene/cameras.py: RGBA 이미지면 알파를 gt_alpha_mask 로 넣는다
      - extract_mesh_tsdf.py: depth[gt_alpha_mask < 0.5] = 0 으로 배경을 버린다
    그래서 알파만 채워주면 학습도 추출도 물체에만 집중한다.

    ⚠ images.txt 의 NAME 을 같이 고쳐야 한다. 3DGS 는 그 이름으로 파일을 찾는다.
    """
    from PIL import Image
    n_done = 0
    ren = {}
    for name, m in masks.items():
        src = os.path.join(img_dir, name)
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        a = m
        if a.shape != (im.size[1], im.size[0]):
            a = np.asarray(Image.fromarray((a.astype(np.uint8) * 255))
                           .resize(im.size, Image.NEAREST)) > 127
        rgba = np.dstack([np.asarray(im),
                          (a.astype(np.uint8) * 255)])
        dst_name = os.path.splitext(name)[0] + ".png"
        Image.fromarray(rgba, "RGBA").save(os.path.join(img_dir, dst_name))
        if dst_name != name:
            os.remove(src)
            ren[name] = dst_name
        n_done += 1

    # 마스크가 없는 프레임은 학습에서 빼는 게 안전하다 — 알파가 없으면
    # 그 프레임만 배경까지 학습해서 TSDF 에 배경이 새어든다.
    dropped = []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".jpg", ".jpeg")) and f not in masks:
            os.remove(os.path.join(img_dir, f))
            dropped.append(f)

    lines = open(images_txt, encoding="utf-8").read().split("\n")
    out = []
    for ln in lines:
        t = ln.split()
        if len(t) >= 10 and not ln.startswith("#"):
            if t[9] in dropped:
                out.append(None)          # 이 이미지 줄과 다음 줄을 뺀다
                continue
            t[9] = ren.get(t[9], t[9])
            ln = " ".join(t)
        out.append(ln)
    # None 바로 뒤의 2D 점 줄까지 제거
    cleaned, skip = [], False
    for ln in out:
        if ln is None:
            skip = True
            continue
        if skip:
            skip = False
            continue
        cleaned.append(ln)
    open(images_txt, "w", encoding="utf-8").write("\n".join(cleaned))
    log(f"  알파 마스크 적용 {n_done}장"
        + (f" · 마스크 없는 {len(dropped)}장 제외" if dropped else ""))
    return n_done, len(dropped)


def sam_filter(xyz, rgb, views, masks, tmp_dir, log=print):
    """SAM 마스크로 초기 점군에서 물체만 남긴다.

    colmap_pipe.filter_by_masks 를 그대로 재사용한다 — 각 점을 모든 뷰에
    투영해 '보이는 뷰 중 마스크 안에 든 비율' 로 거르는 그 로직이다.
    그 함수는 ply 경로를 받고 법선을 읽으므로, 임시 ply 를 거쳐 넘긴다.
    (sparse 점군에는 법선이 없어서 0 으로 채워 넣는다. GOF 는 초기 점군의
     법선을 쓰지 않으므로 문제 없다.)

    재사용하는 또 하나의 이유: 그 함수에는 '남은 점이 2% 미만이면 프롬프트가
    엉뚱한 걸 잡은 것' 이라는 실측 기반 경고가 들어 있다. 직접 짜면 그게 빠진다.
    """
    import colmap_pipe
    import open3d as o3d
    tmp = os.path.join(tmp_dir, "_sam_in.ply")
    store_ply(tmp, xyz, rgb)
    pcd = colmap_pipe.filter_by_masks(tmp, views, masks, log=log)
    try:
        os.remove(tmp)
    except Exception:
        pass
    P = np.asarray(pcd.points, np.float32)
    C = np.asarray(pcd.colors)
    C = (C * 255).astype(np.uint8) if len(C) == len(P) else \
        np.full((len(P), 3), 128, np.uint8)
    return P, C


def run_sam(img_dir, prompt, cache=None, log=print):
    """이미지 폴더에 SAM 실행 → {파일명: bool 마스크}. 없으면 None.

    prompt 가 비어 있으면 sam_mask.segment 가 '중앙 박스' 모드로 간다.

    ⚠ 중앙 박스를 기본으로 삼지 마라. 예전 주석은 '리그 촬영은 물체가 늘 화면
      중앙이라 이름을 모를 때 이쪽이 더 안전하다'고 했는데, 재보니 틀렸다.
      물체는 판 중앙이 아니라 판 위 한쪽에 얹히므로 화면 중앙에 안 온다.
      실측(0831_2058, 115장. 검증된 COLMAP 점군을 투영해 정답으로 삼고,
      그 물체 점을 마스크가 얼마나 덮는지 잰 값):
        stepper motor|the object in the center   113장   덮음 91.6%  ← 최선
        stepper motor                             58장   덮음 93.1%  (절반이 마스크 없음)
        the silver metal object|the metal cylinder 115장  덮음 45.2%  ← 당시 쓰던 것
        (빈 프롬프트 = 중앙 박스)                 115장   덮음  6.7%  ← 사실상 배경
      물체 이름을 주고, 일반 문구를 | 로 덧대라. 이름을 정말 모르겠으면
      'the object in the center' 만이라도 텍스트로 줘라.
    """
    import glob
    import sam_mask
    ok, why = sam_mask.available()
    if not ok:
        log(f"  [SAM 없음] {why} → 마스크 없이 진행")
        return None
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))
                  + glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        return None

    # 마스크는 2분쯤 걸린다. 필터 파라미터를 바꿔가며 다시 볼 일이 많아서 캐시한다.
    if cache and os.path.exists(cache):
        z = np.load(cache)
        have, want = set(z.files), {os.path.basename(p) for p in imgs}
        # ⚠ SfM 은 실행마다 등록 장수가 달라진다(133장 → 117/118장). 캐시가 지금
        # 이미지 목록을 다 덮지 못하면 조용히 일부만 마스킹되므로 버리고 다시 만든다.
        if want <= have:
            log(f"  SAM 캐시 재사용 {len(want)}장")
            return {k: z[k] for k in z.files if k in want}
        log(f"  SAM 캐시가 {len(want - have)}장을 안 덮습니다 → 다시 만듭니다")
    log(f"  SAM {len(imgs)}장 · 프롬프트 {prompt!r}" if prompt
        else f"  SAM {len(imgs)}장 · 중앙 박스 모드")
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
    out = {os.path.basename(p): m for p, m in zip(imgs, masks)
           if m is not None}
    if cache:
        try:
            np.savez_compressed(cache, **out)
            log(f"  SAM 마스크 캐시 저장 → {cache}")
        except Exception as e:
            log(f"  [캐시 실패] {e}")
    return out


# ---------------- COLMAP 경로 ----------------
def prep_colmap(folder, out, max_size=1600, every=1, dense=False,
                ply_override=None, sam=False, sam_prompt="", log=print):
    """사진 → SfM → undistort → GOF 데이터셋.

    dense 융합(patch_match_stereo)은 기본으로 건너뛴다. GOF 는 초기 점군을
    최적화 과정에서 분열·복제시키므로 sparse 로 시작해도 결과가 거의 같고,
    dense 는 19분을 더 쓴다. 굳이 '그때 그 점군'을 넣고 싶을 때만 --dense.
    """
    import colmap_pipe
    cm = colmap_pipe.find_colmap()
    work = os.path.join(out, "_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    log(f"[COLMAP] {cm}")

    img_dir, names = colmap_pipe.collect_images(folder, work, log)
    if every > 1:
        keep = set(names[::every])
        for n in names:
            if n not in keep:
                os.remove(os.path.join(img_dir, n))
        names = [n for n in names if n in keep]
        log(f"  {every}장에 1장씩 추려 {len(names)}장 사용")

    try:
        model = colmap_pipe.sfm(cm, work, img_dir, log)
    except RuntimeError:
        # 실측(0831_2058, 133장): --every 4 로 34장만 남기면 mapper 가
        # "No good initial image pair found" 로 죽는다. 링 촬영은 인접 프레임
        # 사이 각도가 이미 커서, 솎으면 시차가 정합 한계를 넘어간다.
        if every > 1:
            log(f"  [!] {every}장에 1장씩 솎은 게 원인일 수 있습니다 — "
                f"링 촬영은 솎으면 인접 프레임 시차가 너무 커집니다.")
            log(f"      --every 없이(전체 {len(names)*every}장 가까이) 다시 돌려보세요.")
        raise

    # undistort — SIMPLE_RADIAL → PINHOLE. GOF 로더가 PINHOLE 만 받으므로 필수다.
    d = os.path.join(work, "dense")
    t0 = time.time()
    colmap_pipe.run([cm, "image_undistorter", "--image_path", img_dir,
                     "--input_path", model, "--output_path", d,
                     "--output_type", "COLMAP",
                     "--max_image_size", str(max_size)], log)
    txt = os.path.join(d, "sparse_txt")
    os.makedirs(txt, exist_ok=True)
    colmap_pipe.run([cm, "model_converter", "--input_path", os.path.join(d, "sparse"),
                     "--output_path", txt, "--output_type", "TXT"], log)
    log(f"  undistort {time.time()-t0:.0f}s")

    fused = None
    if dense:
        log("  dense 융합 시작 — 수십 분 걸립니다")
        t0 = time.time()
        colmap_pipe.run([cm, "patch_match_stereo", "--workspace_path", d,
                         "--workspace_format", "COLMAP",
                         "--PatchMatchStereo.geom_consistency", "true"], log)
        fused = os.path.join(d, "fused.ply")
        colmap_pipe.run([cm, "stereo_fusion", "--workspace_path", d,
                         "--workspace_format", "COLMAP", "--input_type", "geometric",
                         "--output_path", fused], log)
        log(f"  dense {time.time()-t0:.0f}s")

    # 사진 — undistort 된 것을 쓴다 (cameras.txt 의 PINHOLE 과 짝이 맞는 유일한 버전)
    dst_img = os.path.join(out, "images")
    shutil.rmtree(dst_img, ignore_errors=True)
    shutil.copytree(os.path.join(d, "images"), dst_img)

    sp = os.path.join(out, "sparse", "0")
    os.makedirs(sp, exist_ok=True)
    shutil.copy2(os.path.join(txt, "cameras.txt"), os.path.join(sp, "cameras.txt"))
    shutil.copy2(os.path.join(txt, "images.txt"), os.path.join(sp, "images.txt"))

    src = "colmap-sparse"
    if ply_override:
        import open3d as o3d
        p = o3d.io.read_point_cloud(ply_override)
        xyz = np.asarray(p.points, np.float32)
        rgb = (np.asarray(p.colors) * 255).astype(np.uint8) if p.has_colors() \
            else np.full((len(xyz), 3), 128, np.uint8)
        src = f"ply:{os.path.basename(ply_override)}"
    elif fused:
        import open3d as o3d
        p = o3d.io.read_point_cloud(fused)
        xyz = np.asarray(p.points, np.float32)
        rgb = (np.asarray(p.colors) * 255).astype(np.uint8)
        src = "colmap-dense"
    else:
        xyz, rgb = read_points3D_txt(os.path.join(txt, "points3D.txt"))

    # SAM — 초기 점군에서 물체만 남긴다.
    # 왜 필요한가: 이걸 안 하면 점군에 턴테이블·배경이 그대로 들어가고,
    # gof_pipe 의 clean_mesh 는 '가장 큰 덩어리' 를 남기므로 결과 메쉬가
    # 물체가 아니라 받침 원반이 된다 (실측 0831_2058, 936k 면 중 대부분이 원반).
    # 점군이 좁아지면 TSDF 복셀(점군 크기/512)도 물체 기준으로 잘게 잡힌다.
    n_before = len(xyz)
    if sam:
        masks = run_sam(os.path.join(d, "images"), sam_prompt,
                        cache=os.path.join(out, "sam_masks.npz"), log=log)
        if masks:
            views, _ = colmap_pipe.read_model(txt)
            xyz, rgb = sam_filter(xyz, rgb, views, masks, work, log)
            src += "+sam"
            # 점군만 거르면 GOF 가 배경까지 학습해서 TSDF 추출이 터진다.
            # 이미지에도 마스크를 알파로 구워 넣는다.
            bake_alpha(dst_img, os.path.join(sp, "images.txt"), masks, log)

    ok = bbox_check(xyz, read_cam_centers_txt(os.path.join(sp, "images.txt")), log)
    store_ply(os.path.join(sp, "points3D.ply"), xyz, rgb)
    write_points3D_txt(os.path.join(sp, "points3D.txt"), xyz, rgb)
    log(f"  초기 점군 {len(xyz):,}점 ({src})")

    # ⚠ 넣은 장수가 아니라 '실제로 데이터셋에 들어간' 장수를 센다.
    # SfM 에 등록 못 된 사진은 undistort 결과에도 안 나온다 (실측 133장 → 118장).
    n_used = len([f for f in os.listdir(dst_img)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if n_used < len(names):
        log(f"  [!] {len(names)}장 중 {n_used}장만 정합됐습니다 "
            f"({len(names)-n_used}장 탈락)")
    return dict(n_images=n_used, n_images_input=len(names),
                n_points=int(len(xyz)), n_points_before_sam=int(n_before),
                point_source=src, frame_ok=ok, work=work)


# ---------------- VGGT 경로 ----------------
def prep_vggt(folder, out, res=518, every=1, conf_pct=50, max_points=300_000,
              sam=False, sam_prompt="", log=print):
    """사진 → VGGT 1회 추론 → 포즈 + 점군 → GOF 데이터셋.

    ⚠ 한 번의 추론(단일 윈도우)으로 전 프레임을 처리한다.
    server.py 는 8GB VRAM 때문에 8장씩 끊어 돌리고 윈도우끼리 umeyama 로 이어
    붙이는데(build_groups/align_windows), 그 정합은 스케일 오차를 윈도우 경계마다
    누적시킨다. GOF 는 전 프레임이 '하나의' 좌표계에 있다고 가정하므로 여기서는
    이어붙이기를 아예 안 쓴다. 대신 프레임이 많으면 VRAM 이 터진다 —
    그때는 --every 로 솎거나 --res 를 낮춰라 (아래에서 자동으로 한 번 재시도한다).

    ⚠ 여기서 나온 좌표는 mm 가 아니라 VGGT 임의 스케일이다.
    치수가 필요하면 gof_pipe.py 의 --mm 로 마지막에 곱해라.
    """
    import glob
    import torch
    import torch.nn.functional as F
    from PIL import Image

    cfg_path = os.path.join(os.path.expanduser("~"), "vggt_server", "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    sys.path.insert(0, cfg["vggt_dir"])
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))[::every]
    if len(paths) < 5:
        raise RuntimeError(f"사진이 {len(paths)}장뿐입니다 — 최소 5장")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[VGGT] {dev} · {len(paths)}장 · res {res}")

    model = VGGT.from_pretrained(cfg["model_dir"]).to(dev).eval()

    def infer(r):
        # 비율을 깨지 않고 줄이기만 한다. (세로로 늘려서 mm 치수가 틀어진 전례가 있다)
        imgs = load_and_preprocess_images(paths)
        if r < 518:
            h, w = imgs.shape[-2:]
            sc = r / 518.0
            H = max(14, int(round(h * sc / 14)) * 14)
            W = max(14, int(round(w * sc / 14)) * 14)
            imgs = F.interpolate(imgs, size=(H, W), mode="bilinear", align_corners=False)
        imgs = imgs.to(dev)
        with torch.no_grad():
            if dev == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    p = model(imgs)
            else:
                p = model(imgs)
        return p

    try:
        pred = infer(res)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        res2 = max(154, (int(res * 0.7) // 14) * 14)
        log(f"  [VRAM 부족] res {res} → {res2} 로 한 번 재시도합니다")
        pred = infer(res2)
        res = res2

    wp = pred["world_points"][0].float().cpu().numpy()          # (S,H,W,3)
    cf = pred["world_points_conf"][0].float().cpu().numpy()     # (S,H,W)
    im = pred["images"][0].permute(0, 2, 3, 1).float().cpu().numpy()
    pe = pred["pose_enc"][0].float().cpu().numpy()
    S, H, W = cf.shape
    extr, intr = pose_encoding_to_extri_intri(
        torch.from_numpy(pe)[None].float(), (H, W))
    extr = extr[0].numpy()          # (S,3,4) world→cam, OpenCV = COLMAP 규약
    intr = intr[0].numpy()          # (S,3,3)
    del pred
    if dev == "cuda":
        torch.cuda.empty_cache()

    dst_img = os.path.join(out, "images")
    shutil.rmtree(dst_img, ignore_errors=True)
    os.makedirs(dst_img, exist_ok=True)
    sp = os.path.join(out, "sparse", "0")
    os.makedirs(sp, exist_ok=True)

    # SAM 을 이미지 저장 '전에' 돌린다. 마스크를 알파로 같이 구워야 하기 때문이다.
    # 점군만 걸러서는 부족하다 — GOF 가 배경까지 학습하면 TSDF 추출이 터진다
    # (COLMAP 경로에서 실측: Open3D 가 9.8GB 할당 시도 후 exit 0xC0000005).
    vmasks = None
    if sam:
        import sam_mask
        raw = run_sam(folder, sam_prompt,
                      cache=os.path.join(out, "sam_masks.npz"), log=log)
        if raw:
            names0 = [os.path.basename(p) for p in paths]
            vmasks = [None] * S
            for s in range(S):
                mk = raw.get(names0[s])
                if mk is not None:
                    vmasks[s] = sam_mask.resize_mask(mk, H, W)
            got = sum(1 for v in vmasks if v is not None)
            log(f"  마스크 {got}/{S} 프레임 (VGGT 입력 {W}x{H} 로 축소)")

    cams, views, centers = [], [], []
    for s in range(S):
        name = f"{s:04d}.png"
        # 모델이 실제로 본 그림을 저장한다 — intr 과 해상도가 맞는 유일한 버전이다
        pix = np.clip(im[s] * 255, 0, 255).astype(np.uint8)
        if vmasks is not None and vmasks[s] is not None:
            pix = np.dstack([pix, (vmasks[s].astype(np.uint8) * 255)])
            Image.fromarray(pix, "RGBA").save(os.path.join(dst_img, name))
        else:
            Image.fromarray(pix).save(os.path.join(dst_img, name))
        K = intr[s]
        cams.append(dict(id=s + 1, w=W, h=H, fx=float(K[0, 0]), fy=float(K[1, 1]),
                         cx=float(K[0, 2]), cy=float(K[1, 2])))
        R, t = extr[s][:, :3], extr[s][:, 3]
        views.append(dict(id=s + 1, q=rotmat2qvec(R), t=t, cam_id=s + 1, name=name))
        centers.append(-R.T @ t)

    write_cameras_txt(os.path.join(sp, "cameras.txt"), cams)
    write_images_txt(os.path.join(sp, "images.txt"), views)

    keep = cf >= np.percentile(cf, 100 - conf_pct)

    # 위에서 이미 만든 마스크를 신뢰도 마스크에 AND 로 겹친다 (server.py 와 같은 방식).
    # VGGT 는 픽셀마다 3D 점을 주므로, COLMAP 처럼 투영해서 거를 필요가 없다.
    if vmasks is not None:
        hit = 0
        for s in range(S):
            if vmasks[s] is None:
                continue
            keep[s] &= vmasks[s]
            hit += 1
        log(f"  마스크 적용 {hit}/{S} 프레임 · 남은 픽셀 {int(keep.sum()):,}")
        if keep.sum() < 500:
            log("  [!] 마스크 뒤 남은 픽셀이 너무 적습니다 — 마스크를 버립니다")
            keep = cf >= np.percentile(cf, 100 - conf_pct)

    xyz = wp[keep].reshape(-1, 3).astype(np.float32)
    rgb = np.clip(im[keep].reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
    if len(xyz) > max_points:
        i = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[i], rgb[i]

    ok = bbox_check(xyz, centers, log)
    store_ply(os.path.join(sp, "points3D.ply"), xyz, rgb)
    write_points3D_txt(os.path.join(sp, "points3D.txt"), xyz, rgb)
    log(f"  초기 점군 {len(xyz):,}점 (신뢰도 상위 {conf_pct}%) · 이미지 {W}x{H}")
    return dict(n_images=S, n_points=int(len(xyz)), point_source="vggt",
                frame_ok=ok, res=res, work=None)


def main():
    ap = argparse.ArgumentParser(description="점군+사진 → GOF 입력 데이터셋")
    ap.add_argument("folder")
    ap.add_argument("--pose", choices=["colmap", "vggt"], default="colmap")
    ap.add_argument("--out", default="")
    ap.add_argument("--every", type=int, default=1, help="N장에 1장만 사용")
    ap.add_argument("--dense", action="store_true", help="COLMAP dense 점군으로 초기화(느림)")
    ap.add_argument("--ply", default="", help="초기 점군 교체 — 같은 실행에서 나온 것만")
    ap.add_argument("--sam", action="store_true",
                    help="SAM 으로 초기 점군에서 물체만 남긴다")
    ap.add_argument("--sam-prompt", default="",
                    help="SAM 텍스트 프롬프트. 비우면 중앙 박스 모드(리그 촬영 권장)")
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--res", type=int, default=518, help="VGGT 입력 해상도")
    ap.add_argument("--conf", type=int, default=50, help="VGGT 신뢰도 상위 %%")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()

    folder = os.path.abspath(os.path.expanduser(a.folder.strip().strip('"')))
    if not os.path.isdir(folder):
        print(f"[!] 폴더가 없습니다: {folder}")
        return 1
    out = os.path.abspath(a.out) if a.out else os.path.join(folder, "_gof")
    os.makedirs(out, exist_ok=True)

    print("=" * 62)
    print(f"  {folder}")
    print(f"  포즈 {a.pose} · 결과 → {out}")
    print("=" * 62)

    t0 = time.time()
    if a.pose == "vggt":
        if a.dense or a.ply:
            print("[!] --dense/--ply 는 COLMAP 경로 전용입니다 — 무시합니다")
        rep = prep_vggt(folder, out, res=a.res, every=a.every, conf_pct=a.conf,
                        sam=a.sam, sam_prompt=a.sam_prompt)
    else:
        rep = prep_colmap(folder, out, max_size=a.max_size, every=a.every,
                          dense=a.dense, ply_override=a.ply or None,
                          sam=a.sam, sam_prompt=a.sam_prompt)
        if not a.keep_work and rep.get("work"):
            shutil.rmtree(rep["work"], ignore_errors=True)
            rep["work"] = None      # 지운 폴더를 리포트에 남기지 않는다

    # 쓴 것을 되읽어 검증한다 — 학습 들어가기 전 마지막 관문
    ok, vrep = verify_dataset(out)
    rep["verify"] = vrep
    rep["frame_ok"] = bool(rep.get("frame_ok", True) and ok)

    rep.update(source_folder=folder, pose=a.pose, elapsed_sec=round(time.time() - t0, 1),
               scale_mm=None)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"  이미지 {rep['n_images']}장 · 초기점군 {rep['n_points']:,}점 "
          f"({rep['point_source']}) · {rep['elapsed_sec']:.0f}s")
    if not rep["frame_ok"]:
        print("  [!] 좌표계 점검에 걸렸습니다 — 그대로 학습하면 결과가 망가집니다")
    print(f"  다음: python gof_pipe.py {out}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        sys.exit(1)
