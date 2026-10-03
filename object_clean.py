"""점군에서 물체 하나만 남긴다. 사람 입력 없음.

왜 필요한가
    SAM 마스크가 있어도 서버가 거리 필터를 풀어버리고(SAM_DIST_PCT=100), 마스크를
    못 잡은 프레임의 배경이 그대로 섞인다. 실측(0831_2058, 133장):
        필터 완화       200만점 · 중심거리 최대 4.16 (물체 크기의 17배)
        dist_pct=92     77만점 · 중심거리 최대 4.06   ← 퍼센타일로는 안 잡힌다
    퍼센타일은 '얼마나 버릴지'를 정할 뿐 '무엇을 버릴지'를 모른다. 멀리 뭉쳐 있는
    배경 덩어리는 어느 퍼센타일에서도 살아남는다.

무엇을 쓰는가
    시료는 항상 한 개고 턴테이블 위에 있다. 그러면 '가장 큰 밀도 덩어리'가 곧
    물체다. 밀도 군집(DBSCAN)은 거리가 아니라 연결성을 보므로, 물체에서 떨어져
    있으면 가까이 있어도 버리고 이어져 있으면 멀어도 남긴다.
    실측(같은 점군, 복셀 다운샘플 2만점):
        최대 군집 14,515점(72.5%) 크기 0.90x0.94x0.93  ← 모터
        2위      1,814점( 9.1%) 크기 0.54x1.30x0.51  ← 배경 조각
    eps 를 대각선의 0.4%~1.5%로 바꿔도 최대 군집은 14,515~14,517 로 흔들리지 않았다.
"""
import numpy as np


def clean_object(P, C=None, voxel_frac=0.002, eps_frac=0.004, min_points=10,
                 keep_frac=0.15, log=print):
    """가장 큰 밀도 덩어리만 남긴다. (점, 색) 반환.

    voxel_frac  군집 계산용 다운샘플 크기(대각선 대비). 작을수록 느리고 정밀.
    eps_frac    군집 연결 거리(대각선 대비).
    keep_frac   최대 군집이 이 비율보다 작으면 군집이 갈라진 것으로 보고 포기한다.
                (배경이 물체보다 클 수도 있으므로 조용히 엉뚱한 걸 남기면 안 된다)
    """
    import open3d as o3d

    P = np.asarray(P, np.float64)
    n0 = len(P)
    if n0 < 1000:
        log(f"  [정리 건너뜀] 점이 {n0}개뿐입니다")
        return P, C

    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    if diag <= 0:
        return P, C

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    ds = pc.voxel_down_sample(diag * voxel_frac)
    Q = np.asarray(ds.points)
    if len(Q) < 200:
        log(f"  [정리 건너뜀] 다운샘플이 {len(Q)}점뿐입니다")
        return P, C

    lab = np.array(ds.cluster_dbscan(eps=diag * eps_frac,
                                     min_points=min_points, print_progress=False))
    valid = lab[lab >= 0]
    if not len(valid):
        log("  [정리 건너뜀] 군집을 못 찾았습니다")
        return P, C

    cnt = np.bincount(valid)
    top = int(np.argmax(cnt))
    frac = cnt[top] / len(Q)
    if frac < keep_frac:
        log(f"  [!] 최대 군집이 {100*frac:.1f}% 뿐입니다 — 물체가 갈라졌거나 "
            f"배경이 더 큽니다. 정리하지 않고 넘깁니다")
        return P, C

    core = Q[lab == top]

    # 다운샘플 결과를 원해상도로 되돌린다. 군집 점 주변 반경 안의 원래 점을 살린다.
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(core)))
    rad = diag * eps_frac
    keep = np.zeros(n0, bool)
    for i, p in enumerate(P):
        k, _, _ = tree.search_radius_vector_3d(p, rad)
        if k > 0:
            keep[i] = True

    if keep.sum() < 500:
        log(f"  [정리 건너뜀] 되돌리니 {int(keep.sum())}점뿐입니다")
        return P, C

    Pk = P[keep]
    Ck = None if C is None else np.asarray(C)[keep]
    d1 = float(np.linalg.norm(Pk.max(0) - Pk.min(0)))
    c = np.median(Pk, 0)
    log(f"  [물체 분리] {n0:,} → {int(keep.sum()):,}점 ({100*keep.mean():.1f}%) · "
        f"대각선 {diag:.3f} → {d1:.3f} · "
        f"중심거리 최대 {np.linalg.norm(Pk - c, axis=1).max():.3f}")
    return Pk, Ck


def remove_halo(P, C=None, radius_frac=0.004, keep_pct=50.0, log=print):
    """물체에 붙어 있는 '뿌연 후광'을 밀도로 깎는다.

    왜 군집으로는 안 되는가 — 후광은 물체와 이어져 있어서 DBSCAN 이 같은 덩어리로
    본다. 떼려면 '연결돼 있나'가 아니라 '빽빽한가'를 봐야 한다. 표면은 여러
    프레임이 같은 자리를 맞춰 빽빽하고, 후광은 프레임마다 깊이가 어긋나 생긴
    것이라 성기다.

    실측(0831_2058, 군집 정리 후 72만점, 반경 = 대각선의 0.4%):
        이웃수 분위 1/25/50/90 = 5 / 28 / 99 / 344      ← 확연히 갈린다
        상위 70% → 50.5만점 대각 0.946 중심거리 최대 0.574
        상위 50% → 36.2만점 대각 0.484 최대 0.221       ← 기본값
        상위 30% → 21.7만점 대각 0.437 최대 0.204       (더 깎아도 별 이득 없음)

    ⚠ 얇은 물체(판·철사)는 표면 자체가 성기므로 keep_pct 를 올려야 한다.
      이 리그의 시료는 덩어리라 50 이 무난했다.
    """
    import open3d as o3d
    P = np.asarray(P, np.float64)
    if len(P) < 2000:
        return P, C
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    tree = o3d.geometry.KDTreeFlann(pc)
    rad = diag * radius_frac
    k = np.fromiter((tree.search_radius_vector_3d(p, rad)[0] for p in P),
                    dtype=np.int32, count=len(P))
    th = np.percentile(k, 100.0 - keep_pct)
    sel = k >= th
    if sel.sum() < 1000:
        log(f"  [후광 제거 건너뜀] 남는 점이 {int(sel.sum())}개뿐입니다")
        return P, C
    Q = P[sel]
    log(f"  [후광 제거] {len(P):,} → {int(sel.sum()):,}점 (이웃 {int(th)}개 이상) · "
        f"대각선 {diag:.3f} → {np.linalg.norm(Q.max(0)-Q.min(0)):.3f}")
    return Q, (None if C is None else np.asarray(C)[sel])


def denoise(P, C=None, nb=24, std=2.0, log=print):
    """군집 뒤 남은 산발적 잡음 제거. 표면의 '솜털'을 얇게 깎는다."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(P, np.float64))
    if C is not None:
        Cn = np.asarray(C, np.float64)
        if Cn.max() > 1.5:
            Cn = Cn[:, :3] / 255.0
        pc.colors = o3d.utility.Vector3dVector(Cn[:, :3])
    out, idx = pc.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    idx = np.asarray(idx)
    log(f"  [잡음 제거] {len(P):,} → {len(idx):,}점")
    return np.asarray(out.points), (None if C is None else np.asarray(C)[idx])
