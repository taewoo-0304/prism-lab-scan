"""정렬 실패 원인 진단 — 깊이맵 형상 문제인가, 포즈 문제인가.

vggt_align.py 의 게이트가 떨어졌을 때 쓴다. VGGT 를 한 번만 돌리고 대응점
데이터를 통째로 캐시해서, 이후 분석은 재실행 없이 반복한다(VGGT 추론이 14분).

가르는 방법
    프레임별 어파인 보정 뒤에도 남는 잔차가
      · 이미지 중심에서의 거리와 함께 커진다  → 깊이맵 자체가 접시처럼 휘었다.
        프레임마다 같은 방향이라 a,b 로 흡수되지 않는다. 포즈는 죄가 없다.
      · 반경과 무관하고 프레임마다 통째로 치우친다 → 포즈/스케일 정합 문제.
      · 어느 쪽도 아니고 그냥 산발적       → 깊이 추정 잡음. 프레임을 더 버려야 한다.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# experiments/ 에서 실행해도 상위 폴더의 server·colmap_pipe 를 찾게 한다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vggt_align import read_colmap, fit_affine, fit_scale


def collect(dense_dir, cache, res=518, log=print):
    """VGGT 를 돌려 대응점별 (프레임, 픽셀, 반경, d_colmap, d_vggt) 를 저장."""
    if os.path.exists(cache):
        log(f"캐시 사용 {cache}")
        return dict(np.load(cache))

    import server
    txt = os.path.join(dense_dir, "sparse_txt")
    views, pts = read_colmap(txt)
    paths = [os.path.join(dense_dir, "images", v["name"]) for v in views]
    wp, cf, df, im, pe = server.infer_window(paths, res)
    D = df[..., 0] if df.ndim == 4 else df
    S, H, W = D.shape
    log(f"VGGT {S}장 {H}x{W}")

    fi, uu, vvv, rr, dcs, dvs = [], [], [], [], [], []
    for i, v in enumerate(views):
        c = v["cam"]
        sx, sy = W / c["w"], H / c["h"]
        if not len(v["pid"]):
            continue
        X = np.array([pts[p] for p in v["pid"]])
        dc = (v["R"] @ X.T).T[:, 2] + v["t"][2]
        u = np.round(v["uv"][:, 0]*sx).astype(int)
        vq = np.round(v["uv"][:, 1]*sy).astype(int)
        g = (dc > 1e-6) & (u >= 0) & (u < W) & (vq >= 0) & (vq < H)
        if not g.any():
            continue
        dv = D[i][vq[g], u[g]]
        g2 = np.isfinite(dv) & (dv > 1e-6)
        if not g2.any():
            continue
        # 주점 기준 반경을 이미지 반폭으로 정규화 (0=중심, 1=가장자리)
        rad = np.hypot((v["uv"][g][g2, 0] - c["cx"]) / (c["w"]/2),
                       (v["uv"][g][g2, 1] - c["cy"]) / (c["h"]/2))
        n = int(g2.sum())
        fi.append(np.full(n, i)); rr.append(rad)
        uu.append(u[g][g2]); vvv.append(vq[g][g2])
        dcs.append(dc[g][g2]); dvs.append(dv[g2])

    out = dict(frame=np.concatenate(fi), u=np.concatenate(uu),
               v=np.concatenate(vvv), rad=np.concatenate(rr),
               dc=np.concatenate(dcs), dv=np.concatenate(dvs),
               H=np.array(H), W=np.array(W), S=np.array(S))
    np.savez_compressed(cache, **out)
    log(f"대응점 {len(out['dc']):,}개 캐시 → {cache}")
    return out


def diagnose(z, trunc, log=print):
    fr, rad, dc, dv = z["frame"], z["rad"], z["dc"], z["dv"]
    S = int(z["S"])
    res_all, rad_all, fr_all, a_list = [], [], [], []
    for i in range(S):
        m = fr == i
        if m.sum() < 30:
            continue
        a, b, keep = fit_affine(dv[m], dc[m])
        r = dc[m] - (a*dv[m] + b)
        res_all.append(r); rad_all.append(rad[m]); fr_all.append(np.full(m.sum(), i))
        a_list.append(a)
    R = np.concatenate(res_all); RA = np.concatenate(rad_all)
    FR = np.concatenate(fr_all); A = np.array(a_list)

    log(f"\n대응점 {len(R):,}개 · 프레임 {len(A)}개 · 절단 {trunc:.5f}")
    log(f"|잔차| 중앙 {np.median(np.abs(R)):.5f} · p90 {np.percentile(np.abs(R),90):.5f}")

    log("\n[A] 반경별 잔차 — 커지면 깊이맵이 휜 것")
    edges = [0, .25, .5, .75, 1.0, 1.5]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (RA >= lo) & (RA < hi)
        if m.sum() < 50:
            continue
        log(f"  반경 {lo:.2f}~{hi:.2f}  n={m.sum():6d}  "
            f"|잔차| 중앙 {np.median(np.abs(R[m])):.5f}  "
            f"부호있는 중앙 {np.median(R[m]):+.5f}  "
            f"절단대비 {np.median(np.abs(R[m]))/trunc:.2f}배")
    cr = float(np.corrcoef(RA, np.abs(R))[0, 1])
    log(f"  반경 vs |잔차| 상관계수 {cr:+.3f}")

    log("\n[B] 프레임 편향 — 크면 포즈/스케일 정합 문제")
    bias = np.array([np.median(R[FR == i]) for i in np.unique(FR)])
    log(f"  프레임별 잔차중앙의 표준편차 {bias.std():.5f} "
        f"(절단의 {bias.std()/trunc:.2f}배)")
    log(f"  어파인 후에도 프레임이 통째로 치우친 정도: "
        f"|중앙| 최대 {np.abs(bias).max():.5f}")

    log("\n[C] 프레임 내부 산포 — 크면 깊이 잡음")
    spread = np.array([np.percentile(np.abs(R[FR == i] - np.median(R[FR == i])), 90)
                       for i in np.unique(FR)])
    log(f"  프레임내 p90 편차의 중앙 {np.median(spread):.5f} "
        f"(절단의 {np.median(spread)/trunc:.2f}배)")

    log("\n[D] 스케일 a 분포")
    log(f"  중앙 {np.median(A):.4f} · std/중앙 {100*A.std()/np.median(A):.2f}% · "
        f"p5~p95 {np.percentile(A,5):.3f}~{np.percentile(A,95):.3f}")
    return dict(radial_corr=cr, frame_bias_std=float(bias.std()),
                inframe_p90=float(np.median(spread)), trunc=trunc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dense")
    ap.add_argument("--cache", default="")
    ap.add_argument("--trunc", type=float, required=True)
    a = ap.parse_args()
    cache = a.cache or os.path.join(a.dense, "_align_cache.npz")
    z = collect(os.path.abspath(a.dense), cache)
    diagnose(z, a.trunc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
