"""VGGT 깊이맵을 COLMAP sparse 점으로 프레임별 보정한다 (융합은 하지 않는다).

문제
    VGGT 깊이맵을 그대로 TSDF 로 융합하면 형상이 안 나온다. 실측(0831_2058,
    133장): 물체 크기 9.25, 복셀 0.0241, 절단 0.0963 인데 프레임별 깊이
    스케일이 약 3.9% 흔들린다 → 0.36, 절단의 4배다. 프레임 A 가 그린 표면과
    프레임 B 가 그린 표면이 다른 복셀에 떨어져서 융합이 안 되고 어긋난 껍질만
    쌓인다.

접근
    COLMAP sparse 는 프레임 간에 일관된 3D 점과 그 점이 각 이미지의 어느
    픽셀에서 관측됐는지를 갖고 있다. 그 픽셀에서 VGGT 깊이를 읽어
        d_colmap ≈ a_i * d_vggt + b_i
    를 프레임마다 맞추면, 모든 프레임이 COLMAP 의 한 스케일로 모인다.

    ⚠ 이건 프레임 간 불일치만 고친다. VGGT 깊이맵 자체의 접시 왜곡(중심에서
      가장자리로 휘는 것)은 프레임마다 같은 방향이라 a,b 로 흡수되지 않는다.
      기존 측정: 프레임별 스케일 보정으로 평면 잔차 4.22% → 3.99% 였다.

    ⚠ 융합은 COLMAP 의 포즈·내부파라미터로 해야 한다. 깊이만 COLMAP 스케일로
      바꾸고 VGGT 포즈로 융합하면 깊이와 extrinsic 의 단위가 달라 더 나빠진다.

이 파일은 진단까지만 한다. 잔차가 절단 거리보다 충분히 작다는 게 숫자로
확인되기 전에는 융합을 돌리지 않는다 — 그게 이 작업의 요점이다.

    python vggt_align.py <dense폴더>          # dense/images 와 dense/sparse_txt 가 있는 곳
"""
import argparse, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# experiments/ 에서 실행해도 상위 폴더의 server·colmap_pipe 를 찾게 한다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------- COLMAP 읽기 ----------------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])


def read_colmap(txt_dir):
    """(뷰 목록, 3D점 dict) 반환. 뷰마다 관측 픽셀과 3D 점 ID 를 함께 담는다."""
    cams = {}
    for line in open(os.path.join(txt_dir, "cameras.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        p = [float(x) for x in t[4:]]
        cams[int(t[0])] = dict(w=int(t[2]), h=int(t[3]),
                               fx=p[0], fy=p[1], cx=p[2], cy=p[3])

    pts = {}
    for line in open(os.path.join(txt_dir, "points3D.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        pts[int(t[0])] = np.array([float(t[1]), float(t[2]), float(t[3])])

    lines = [l for l in open(os.path.join(txt_dir, "images.txt"), encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    views = []
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        obs = lines[i+1].split() if i+1 < len(lines) else []
        uv, pid = [], []
        for j in range(0, len(obs), 3):
            p3 = int(obs[j+2])
            if p3 == -1:
                continue
            uv.append((float(obs[j]), float(obs[j+1])))
            pid.append(p3)
        views.append(dict(R=qvec2rotmat([float(x) for x in t[1:5]]),
                          t=np.array([float(x) for x in t[5:8]]),
                          cam=cams[int(t[8])], name=t[9],
                          uv=np.array(uv, float).reshape(-1, 2),
                          pid=np.array(pid, int)))
    return views, pts


# ---------------- 적합 ----------------
def fit_scale(dv, dc, iters=5, k=2.5):
    """d_c ≈ a * d_v. 반복적 이상치 제거(잔차 k*MAD 밖을 버림)."""
    m = np.ones(len(dv), bool)
    a = 1.0
    for _ in range(iters):
        if m.sum() < 8:
            break
        a = float(np.sum(dv[m]*dc[m]) / max(np.sum(dv[m]*dv[m]), 1e-12))
        r = dc - a*dv
        s = 1.4826*np.median(np.abs(r[m] - np.median(r[m]))) + 1e-12
        m = np.abs(r - np.median(r[m])) < k*s
    return a, 0.0, m


def fit_affine(dv, dc, iters=5, k=2.5):
    """d_c ≈ a * d_v + b. 같은 방식의 반복 이상치 제거."""
    m = np.ones(len(dv), bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        if m.sum() < 8:
            break
        A = np.stack([dv[m], np.ones(m.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, dc[m], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = dc - (a*dv + b)
        s = 1.4826*np.median(np.abs(r[m] - np.median(r[m]))) + 1e-12
        m = np.abs(r - np.median(r[m])) < k*s
    return a, b, m


def analyse(dense_dir, res=518, min_corr=30, log=print):
    """VGGT 를 돌리고 프레임별로 보정계수와 잔차를 낸다. 융합은 하지 않는다."""
    import server                      # import 시점에 VGGT 가 GPU 로 올라간다

    txt = os.path.join(dense_dir, "sparse_txt")
    img_dir = os.path.join(dense_dir, "images")
    views, pts = read_colmap(txt)
    log(f"COLMAP 뷰 {len(views)}개 · 3D 점 {len(pts):,}개")

    names = [v["name"] for v in views]
    paths = [os.path.join(img_dir, n) for n in names]
    missing = [n for n, p in zip(names, paths) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"이미지가 없습니다: {missing[:3]}")

    wp, cf, df, im, pe = server.infer_window(paths, res)
    D = df[..., 0] if df.ndim == 4 else df
    S, H, W = D.shape
    log(f"VGGT 깊이맵 {S}장 · {H}x{W}")
    if S != len(views):
        raise RuntimeError(f"프레임 수 불일치: VGGT {S} vs COLMAP {len(views)}")

    rows = []
    for i, v in enumerate(views):
        c = v["cam"]
        # 전처리는 단순 리사이즈다(crop 모드, 세로가 518 미만이라 잘림 없음).
        sx, sy = W / c["w"], H / c["h"]
        if len(v["pid"]) < min_corr:
            rows.append(dict(i=i, name=v["name"], n=0, ok=False))
            continue

        X = np.array([pts[p] for p in v["pid"]])
        dc = (v["R"] @ X.T).T[:, 2] + v["t"][2]          # COLMAP 카메라 깊이
        u = np.round(v["uv"][:, 0]*sx).astype(int)
        vv = np.round(v["uv"][:, 1]*sy).astype(int)
        good = (dc > 1e-6) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        dv = np.full(len(dc), np.nan)
        dv[good] = D[i][vv[good], u[good]]
        good &= np.isfinite(dv) & (dv > 1e-6)
        if good.sum() < min_corr:
            rows.append(dict(i=i, name=v["name"], n=int(good.sum()), ok=False))
            continue

        a_s, b_s, m_s = fit_scale(dv[good], dc[good])
        a_a, b_a, m_a = fit_affine(dv[good], dc[good])
        r_s = np.abs(dc[good] - a_s*dv[good])
        r_a = np.abs(dc[good] - (a_a*dv[good] + b_a))
        rows.append(dict(
            i=i, name=v["name"], n=int(good.sum()), ok=True,
            scale_a=a_s, scale_med=float(np.median(r_s)),
            scale_p90=float(np.percentile(r_s, 90)),
            aff_a=a_a, aff_b=b_a, aff_med=float(np.median(r_a)),
            aff_p90=float(np.percentile(r_a, 90)),
            raw_med=float(np.median(np.abs(dc[good] - dv[good])))))

    return dict(rows=rows, D=D, views=views, wp=wp, cf=cf, im=im, S=S, H=H, W=W)


def report(rows, log=print):
    ok = [r for r in rows if r["ok"]]
    log(f"\n적합 성공 {len(ok)}/{len(rows)} 프레임")
    if not ok:
        return None
    A = np.array([r["scale_a"] for r in ok])
    log(f"\n프레임별 스케일 a  (d_colmap = a x d_vggt)")
    log(f"  중앙 {np.median(A):.5f} · 최소 {A.min():.5f} · 최대 {A.max():.5f}")
    log(f"  퍼짐 (최대-최소)/중앙 = {100*(A.max()-A.min())/np.median(A):.2f}%")
    log(f"  표준편차/중앙          = {100*A.std()/np.median(A):.2f}%   "
        f"← 이게 프레임 간 불일치")
    for tag, mk, pk in [("스케일만", "scale_med", "scale_p90"),
                        ("어파인  ", "aff_med", "aff_p90")]:
        M = np.array([r[mk] for r in ok]); P = np.array([r[pk] for r in ok])
        log(f"\n  {tag} 잔차 (COLMAP 단위)")
        log(f"    프레임별 중앙값의 중앙 {np.median(M):.5f} · 최대 {M.max():.5f}")
        log(f"    프레임별 p90    의 중앙 {np.median(P):.5f} · 최대 {P.max():.5f}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="VGGT 깊이 ↔ COLMAP sparse 정렬 진단")
    ap.add_argument("dense", help="dense 폴더 (images/ 와 sparse_txt/ 가 있는 곳)")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = analyse(os.path.abspath(a.dense), a.res)
    ok = report(r["rows"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in row.items()} for row in r["rows"]],
                      f, ensure_ascii=False, indent=2)
        print(f"\n프레임별 수치 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
