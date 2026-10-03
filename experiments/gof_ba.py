"""VGGT 포즈를 COLMAP 번들 조정(BA)으로 다듬는다.

    python gof_ba.py <gof_prep 이 만든 _gof 폴더>

왜 필요한가
  VGGT 포즈는 GOF/3DGS 에 그대로 쓰기엔 정밀도가 부족하다. 실측(0831_2058, 133장,
  COLMAP 포즈 기준):

      카메라 위치 잔차 중앙값 2.43%
      카메라 방향 오차 중앙값 3.46도

  3DGS 계열은 서브픽셀 포즈를 전제한다. 518px 폭·화각 55도면 3.46도는 화면상
  약 30픽셀이다. 가우시안이 수렴하지 못하고 뭉개진다 — 학습에 쓴 사진을 다시
  렌더한 PSNR 이 21.0 dB 밖에 안 나왔다(정상이면 30~40).

무엇을 하는가
  VGGT 포즈를 '초기값' 으로 넣고 COLMAP 에 삼각측량 + BA 를 시킨다.

      feature_extractor    특징점 추출
      matcher              매칭
      point_triangulator   주어진 포즈로 3D 점 삼각측량
      bundle_adjuster      포즈와 점을 함께 최적화

  SfM 의 '초기 페어 찾기 + 점진적 등록' 을 건너뛴다. 그래서 COLMAP SfM 이
  조각나는 데이터셋에서도 전 프레임을 한 좌표계로 유지할 수 있다
  (0831_2058 은 COLMAP SfM 이 모델 2개로 쪼개져 133장 중 116장만 건졌다).

결과는 <폴더>/sparse/0 을 덮어쓴다. 원본은 sparse/0_vggt 로 백업한다.
"""
import os, sys, json, time, shutil, argparse
import numpy as np

# 콘솔이 cp949 면 죽는다 (.bat 는 chcp 65001 을 걸어준다)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# experiments/ 에서 실행해도 상위 폴더의 server·colmap_pipe 를 찾게 한다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_model_txt(sp):
    """cameras.txt / images.txt 를 읽는다. (빈 줄 유무에 상관없이 동작)"""
    cams = {}
    for line in open(os.path.join(sp, "cameras.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = dict(model=t[1], w=int(t[2]), h=int(t[3]),
                               params=[float(x) for x in t[4:]])
    imgs = []
    lines = [l for l in open(os.path.join(sp, "images.txt"), encoding="utf-8")
             if not l.startswith("#")]
    for ln in lines:
        t = ln.split()
        if len(t) >= 10 and t[0].isdigit():
            imgs.append(dict(id=int(t[0]), q=[float(x) for x in t[1:5]],
                             tv=[float(x) for x in t[5:8]],
                             cam_id=int(t[8]), name=t[9]))
    return cams, imgs


def write_prior_model(out_dir, cams, imgs, log=print):
    """point_triangulator 에 넣을 '포즈만 있는' 모델을 쓴다.

    ⚠ 둘째 줄(2D 점 목록)은 반드시 '빈 줄' 이어야 한다.
    gof_prep 은 자기 파서 호환 때문에 더미 관측 "0 0 -1" 을 넣는데,
    그대로 두면 point_triangulator 가 존재하지 않는 3D 점을 참조하는
    관측으로 읽어 삼각측량이 어긋난다. 여기서는 COLMAP 규약대로 비운다.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# Camera list\n")
        for cid, c in sorted(cams.items()):
            f.write(f"{cid} {c['model']} {c['w']} {c['h']} "
                    + " ".join(f"{v:.10f}" for v in c["params"]) + "\n")
    with open(os.path.join(out_dir, "images.txt"), "w", encoding="utf-8") as f:
        f.write("# Image list\n")
        for im in imgs:
            q, tv = im["q"], im["tv"]
            f.write(f"{im['id']} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{tv[0]:.10f} {tv[1]:.10f} {tv[2]:.10f} "
                    f"{im['cam_id']} {im['name']}\n")
            f.write("\n")                      # 관측 없음
    open(os.path.join(out_dir, "points3D.txt"), "w", encoding="utf-8").write(
        "# 3D point list\n")
    log(f"  초기 모델 {len(imgs)}장 · 카메라 {len(cams)}개")


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def rot2qvec(R):
    """gof_prep 의 검증된 구현을 그대로 쓴다.

    ⚠ 직접 다시 쓰지 마라. 손으로 옮겨 적었다가 마지막 행 부호를 뒤집어
    회전 왕복 오차가 1.89(정상 1e-16)가 났고, BA 결과 방향 오차가
    3.46도 -> 87.79도로 폭발했다.
    """
    import gof_prep
    return gof_prep.rotmat2qvec(R)


def umeyama(A, B):
    """A -> B 로 보내는 similarity (s, R, t)."""
    ma, mb = A.mean(0), B.mean(0)
    X, Y = A - ma, B - mb
    H = X.T @ Y / len(A)
    U, D, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    s = float((D * np.array([1, 1, d])).sum() / ((X ** 2).sum() / len(A)))
    return s, R, mb - s * R @ ma


def run_all(ds, matcher="sequential", overlap=20, keep_work=False,
            src_images=None, rounds=4, fit_intrinsics=False, log=print):
    import colmap_pipe
    ds = os.path.abspath(os.path.expanduser(ds))
    sp = os.path.join(ds, "sparse", "0")
    if not os.path.isdir(sp):
        raise RuntimeError(f"GOF 데이터셋이 아닙니다: {ds}")
    cm = colmap_pipe.find_colmap()
    log(f"[COLMAP] {cm}")

    work = os.path.join(ds, "_ba")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "database.db")

    cams, imgs = read_model_txt(sp)

    # ⚠ BA 는 데이터셋 이미지가 아니라 '원본 고해상도 사진' 으로 돌려야 한다.
    # VGGT 데이터셋 이미지는 518x294 라 특징점이 이미지당 105개밖에 안 잡히고
    # (정상은 수천 개), 검은 드럼은 무텍스처라 더 나쁘다. 실측: 3D 점 1,197개,
    # 이미지당 관측 29.6개로 BA 가 사실상 아무것도 못 고쳤다.
    # 포즈(R,t)는 해상도와 무관하므로 원본으로 BA 한 결과를 그대로 쓸 수 있다.
    name_map = None
    if src_images:
        import glob
        from PIL import Image
        srcs = sorted(glob.glob(os.path.join(src_images, "*.jpg"))
                      + glob.glob(os.path.join(src_images, "*.jpeg")))
        if len(srcs) < len(imgs):
            raise RuntimeError(f"원본 사진 {len(srcs)}장 < 데이터셋 {len(imgs)}장")
        W2, H2 = Image.open(srcs[0]).size
        # 데이터셋 이름 0000.png 는 sorted(원본)[0] 에 대응한다 (gof_prep 규약)
        name_map = {}
        for im in imgs:
            idx = int(os.path.splitext(im["name"])[0])
            name_map[im["name"]] = os.path.basename(srcs[idx])
            im["name"] = name_map[im["name"]]
        c0 = cams[imgs[0]["cam_id"]]
        f0 = c0["params"][0] * W2 / c0["w"]
        # 카메라 하나를 공유시킨다 — 실제로 같은 카메라이고, 133개로 쪼개는 것보다
        # 조건수가 훨씬 좋다. 왜곡은 SIMPLE_RADIAL 로 BA 가 흡수하게 둔다.
        cams = {1: dict(model="SIMPLE_RADIAL", w=W2, h=H2,
                        params=[f0, W2 / 2.0, H2 / 2.0, 0.0])}
        for im in imgs:
            im["cam_id"] = 1
        img_dir = src_images
        log(f"  BA 입력: 원본 {W2}x{H2} · 초기 초점거리 {f0:.1f} (VGGT {c0['w']}x{c0['h']} 에서 환산)")
    else:
        img_dir = os.path.join(ds, "images")

    prior = os.path.join(work, "prior")
    write_prior_model(prior, cams, imgs, log)

    t0 = time.time()
    log("\n[1/4] 특징점 추출")
    colmap_pipe.run([cm, "feature_extractor", "--database_path", db,
                     "--image_path", img_dir,
                     "--ImageReader.camera_model",
                     "SIMPLE_RADIAL" if src_images else "PINHOLE",
                     "--ImageReader.single_camera", "1" if src_images else "0"], log)

    log(f"[2/4] 매칭 ({matcher})")
    if matcher == "sequential":
        # 링 촬영은 이웃 프레임끼리 겹친다. exhaustive 보다 훨씬 빠르고,
        # loop_detection 으로 한 바퀴 돈 지점도 이어준다.
        colmap_pipe.run([cm, "sequential_matcher", "--database_path", db,
                         "--SequentialMatching.overlap", str(overlap),
                         "--SequentialMatching.loop_detection", "0"], log)
    else:
        colmap_pipe.run([cm, "exhaustive_matcher", "--database_path", db], log)

    # ⚠ 삼각측량 임계값을 한 번에 기본값으로 쓰면 안 된다.
    # tri_create_max_angle_error 기본값은 2도인데 VGGT 포즈 오차가 3.46도다.
    # 임계값보다 오차가 커서 대응점 대부분이 버려지고(실측 3,781점 · 이미지당
    # 73관측), 그 얇은 구속으로 BA 를 돌리면 포즈가 오히려 나빠졌다(3.46 -> 4.74도).
    # 느슨하게 시작해 조여가며 삼각측량↔BA 를 반복한다 — 표준 부트스트랩이다.
    schedule = [(12.0, 24.0), (6.0, 12.0), (3.0, 6.0), (2.0, 4.0)][:max(1, rounds)]
    cur = prior
    _A0 = np.array([-qvec2rot(x["q"]).T @ np.array(x["tv"]) for x in imgs])
    ext0 = float((_A0.max(0) - _A0.min(0)).max())   # 발산 감시 기준
    for r, (ang_err, reproj) in enumerate(schedule, 1):
        tri = os.path.join(work, f"tri{r}")
        ba = os.path.join(work, f"ba{r}")
        os.makedirs(tri, exist_ok=True); os.makedirs(ba, exist_ok=True)
        log(f"[3/4] 라운드 {r}/{len(schedule)} 삼각측량 "
            f"(각도오차 {ang_err}도 · 재투영 {reproj}px)")
        colmap_pipe.run([cm, "point_triangulator", "--database_path", db,
                         "--image_path", img_dir,
                         "--input_path", cur, "--output_path", tri,
                         "--Mapper.tri_create_max_angle_error", str(ang_err),
                         "--Mapper.tri_continue_max_angle_error", str(ang_err),
                         "--Mapper.tri_merge_max_reproj_error", str(reproj),
                         "--Mapper.tri_complete_max_reproj_error", str(reproj),
                         "--Mapper.filter_max_reproj_error", str(reproj)], log)
        log(f"[4/4] 라운드 {r} 번들 조정")
        # ⚠ 내부 파라미터를 자유롭게 두면 BA 가 포즈와 초점거리·왜곡을 서로
        # 맞바꾸며 헤맨다. 대조군(COLMAP 자기 포즈로 시작)에서 방향 오차가
        # 1.08도나 벌어졌다 — 보존돼야 정상인데.
        # 초점거리와 왜곡을 고정하고 포즈·점만 최적화한다.
        colmap_pipe.run([cm, "bundle_adjuster", "--input_path", tri,
                         "--output_path", ba,
                         "--BundleAdjustment.refine_principal_point", "0",
                         "--BundleAdjustment.refine_focal_length",
                         "1" if fit_intrinsics else "0",
                         "--BundleAdjustment.refine_extra_params",
                         "1" if fit_intrinsics else "0"], log)
        # 다음 라운드는 이번 BA 결과를 초기값으로 쓴다
        rt = os.path.join(work, f"ba{r}_txt")
        os.makedirs(rt, exist_ok=True)
        colmap_pipe.run([cm, "model_converter", "--input_path", ba,
                         "--output_path", rt, "--output_type", "TXT"], log)
        rc, ri = read_model_txt(rt)
        npts = sum(1 for l in open(os.path.join(rt, "points3D.txt"), encoding="utf-8")
                   if not l.startswith("#") and l.strip())
        A = np.array([-qvec2rot(x["q"]).T @ np.array(x["tv"]) for x in ri])
        ext = float((A.max(0) - A.min(0)).max())
        log(f"    → {len(ri)}장 · 3D 점 {npts:,} · 카메라 배치 크기 {ext:.2f}")

        # ⚠ 다음 라운드에 BA 출력을 '그대로' 넣으면 안 된다.
        # 그 모델엔 이미 3D 점과 관측이 들어 있어서, 거기에 또 삼각측량을 얹으면
        # 관측이 중복 누적되며 퇴화한다. 실측(대조군, COLMAP 의 정상 포즈로 시작):
        #   prior 9.11 → 라운드1 9.13 → 라운드2 22.8 → 라운드3 191,242 → 라운드4 4.6e7
        # 라운드 1 은 멀쩡한데 2부터 발산했다. 포즈만 남기고 점은 비워서 넘긴다.
        nxt = os.path.join(work, f"prior{r+1}")
        write_prior_model(nxt, rc, ri, log=lambda *a: None)
        cur = nxt

        if ext > ext0 * 5:
            log(f"    [!] 카메라 배치가 처음({ext0:.2f})의 5배를 넘었습니다 — "
                f"발산으로 보고 라운드 {r} 에서 멈춥니다")
            ba = os.path.join(work, f"ba{r}")
            break
        ba = os.path.join(work, f"ba{r}")

    txt = os.path.join(work, "ba_txt")
    os.makedirs(txt, exist_ok=True)
    colmap_pipe.run([cm, "model_converter", "--input_path", ba,
                     "--output_path", txt, "--output_type", "TXT"], log)
    log(f"  BA 완료 {time.time()-t0:.0f}s")

    # 결과 반영 — 원본은 남긴다
    bak = os.path.join(ds, "sparse", "0_vggt")
    if not os.path.exists(bak):
        shutil.copytree(sp, bak)
        log(f"  원본 포즈 백업 → {bak}")

    _, ba_imgs = read_model_txt(txt)
    log(f"  BA 결과 {len(ba_imgs)}/{len(imgs)}장")
    if len(ba_imgs) < len(imgs):
        log(f"  [!] {len(imgs)-len(ba_imgs)}장이 BA 에서 빠졌습니다")

    if not src_images:
        for f in ("cameras.txt", "images.txt"):
            shutil.copy2(os.path.join(txt, f), os.path.join(sp, f))
        return dict(n_in=len(imgs), n_out=len(ba_imgs),
                    sec=round(time.time() - t0, 1))

    # ⚠ BA 는 좌표계(gauge)를 자유롭게 바꾼다. 그대로 쓰면 VGGT 초기 점군이
    # 카메라와 어긋난다. BA 포즈를 VGGT 좌표계로 되돌리는 similarity 를 구해
    # 적용한다 — BA 가 고친 '상대적' 개선은 유지되고 전역 배율만 복원된다.
    old_cams, old_imgs = read_model_txt(bak)
    old_c = {im["name"]: -qvec2rot(im["q"]).T @ np.array(im["tv"]) for im in old_imgs}
    rev = {v: k for k, v in name_map.items()}
    A, B, keep = [], [], []
    for im in ba_imgs:
        ds_name = rev.get(im["name"])
        if ds_name is None or ds_name not in old_c:
            continue
        A.append(-qvec2rot(im["q"]).T @ np.array(im["tv"]))
        B.append(old_c[ds_name])
        keep.append((im, ds_name))
    s, R, t = umeyama(np.array(A), np.array(B))
    log(f"  BA→VGGT 좌표계 복원: 배율 {s:.4f}")

    out_imgs = []
    for im, ds_name in keep:
        Rw = qvec2rot(im["q"])            # world->cam
        C = -Rw.T @ np.array(im["tv"])    # 카메라 중심
        C2 = s * (R @ C) + t
        Rw2 = Rw @ R.T                    # 회전도 같이 돌린다
        out_imgs.append(dict(id=len(out_imgs) + 1, q=rot2qvec(Rw2),
                             tv=-Rw2 @ C2, cam_id=1, name=ds_name))
    # 내부 파라미터는 원래 VGGT 것(518x294)을 그대로 쓴다 — 학습 이미지가 그 크기다.
    # ⚠ 최종 출력은 write_prior_model 로 쓰면 안 된다. 그건 둘째 줄을 빈 줄로 쓰는데,
    # colmap_pipe.read_model 이 빈 줄을 걸러낸 뒤 2줄씩 끊어서 뷰가 절반이 된다
    # (실제로 이 함정에 두 번 걸렸다). gof_prep 의 검증된 writer 를 쓴다.
    import gof_prep
    c0 = old_cams[old_imgs[0]["cam_id"]]
    gof_prep.write_cameras_txt(
        os.path.join(sp, "cameras.txt"),
        [dict(id=1, w=c0["w"], h=c0["h"], fx=c0["params"][0], fy=c0["params"][1],
              cx=c0["params"][2], cy=c0["params"][3])])
    gof_prep.write_images_txt(
        os.path.join(sp, "images.txt"),
        [dict(id=im["id"], q=im["q"], t=im["tv"], cam_id=1, name=im["name"])
         for im in out_imgs])
    log(f"  포즈 갱신 {len(out_imgs)}장 (내부 파라미터는 VGGT 값 유지)")

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return dict(n_in=len(imgs), n_out=len(out_imgs), sec=round(time.time()-t0, 1))


def main():
    ap = argparse.ArgumentParser(description="VGGT 포즈 → COLMAP BA 로 정밀화")
    ap.add_argument("dataset")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"],
                    default="sequential")
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--fit-intrinsics", action="store_true",
                    help="BA 가 초점거리·왜곡도 함께 조정 (기본은 고정)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="삼각측량↔BA 반복 횟수 (임계값을 점점 조인다)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--src-images", default="",
                    help="BA 에 쓸 원본 고해상도 사진 폴더. 강력 권장 — "
                         "데이터셋 이미지(518x294)로는 특징점이 안 잡힌다")
    a = ap.parse_args()
    try:
        rep = run_all(a.dataset, matcher=a.matcher, overlap=a.overlap,
                      keep_work=a.keep_work, src_images=a.src_images or None,
                      rounds=a.rounds, fit_intrinsics=a.fit_intrinsics)
    except Exception as e:
        print(f"\n[실패] {e}")
        return 1
    print("\n" + "=" * 62)
    print(f"  {rep['n_in']}장 → BA 후 {rep['n_out']}장 · {rep['sec']:.0f}s")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
