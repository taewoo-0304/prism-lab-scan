"""실루엣 카빙 — 마스크와 카메라 포즈만으로 닫힌 메쉬를 만든다.

왜 이게 필요한가
    VGGT 깊이맵은 한 프레임 안에서도 실제 형상과 TSDF 절단거리의 3.13배씩
    어긋난다(2026-09-11 측정). 마스크를 133/133 완벽하게 잡고 배경을 다 걷어내도
    메쉬가 안장처럼 휘었다. 깊이 추정을 쓰는 한 이건 안 고쳐진다.
    COLMAP 덴스 MVS 는 제대로 된 깊이를 주지만 patch_match 에만 40~60분이 든다.

    실루엣 카빙은 깊이를 아예 안 쓴다. '이 복셀이 어느 뷰에서든 마스크 밖에
    있으면 물체가 아니다' 만 반복한다. 필요한 건 마스크와 포즈뿐이고, 둘 다
    이미 싸게 얻는다 — 마스크는 SAM 이 60초, 포즈는 COLMAP SfM 이 103초.

한계 (정직하게)
    오목한 부분은 못 판다. 실루엣만 보므로 구멍·홈·파인 곳은 메워진 채 나온다.
    바깥 형상과 전체 치수는 맞고, 결과가 항상 닫혀 있어서 3D 프린트에 바로 쓴다.
    반사가 심해 마스크가 몇 프레임 틀려도 `keep_frac` 이 흡수한다.
"""
import numpy as np


def _project(P, v):
    """월드 점 → 픽셀 (u, v) 와 '앞에 있나' 플래그."""
    c = v["cam"]
    X = (v["R"] @ P.T).T + v["t"]
    z = X[:, 2]
    ok = z > 1e-6
    u = np.full(len(P), -1.0, np.float32)
    w = np.full(len(P), -1.0, np.float32)
    u[ok] = c["fx"] * X[ok, 0] / z[ok] + c["cx"]
    w[ok] = c["fy"] * X[ok, 1] / z[ok] + c["cy"]
    return u, w, ok


def _count_inside(P, views, masks, chunk=2_000_000):
    """각 점이 '마스크 안'인 뷰 수와 '화면 안'인 뷰 수를 센다."""
    n = len(P)
    inside = np.zeros(n, np.int16)
    seen = np.zeros(n, np.int16)
    for s in range(0, n, chunk):
        Q = P[s:s + chunk]
        ins = np.zeros(len(Q), np.int16)
        see = np.zeros(len(Q), np.int16)
        for v in views:
            m = masks.get(v["name"])
            if m is None:
                continue
            c = v["cam"]
            u, w, ok = _project(Q, v)
            inb = ok & (u >= 0) & (u < c["w"]) & (w >= 0) & (w < c["h"])
            see += inb
            idx = np.where(inb)[0]
            if len(idx):
                ins[idx] += m[w[idx].astype(np.int32), u[idx].astype(np.int32)]
        inside[s:s + chunk] = ins
        seen[s:s + chunk] = see
    return inside, seen


def _grid(lo, hi, n):
    ax = [np.linspace(lo[i], hi[i], n, dtype=np.float32) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return g, ax


def carve(views, masks, lo, hi, res=256, keep_frac=0.92, log=print):
    """상자 [lo, hi] 안을 res^3 으로 깎는다. (occupancy 3D 배열, 축) 반환.

    keep_frac  이 비율 이상의 '본 뷰'에서 마스크 안이어야 물체로 남긴다.
               1.0 으로 두면 마스크가 한 프레임만 틀려도 구멍이 뚫린다.
    """
    P, ax = _grid(lo, hi, res)
    inside, seen = _count_inside(P, views, masks)
    ratio = np.where(seen > 0, inside / np.maximum(seen, 1), 0.0)
    occ = (ratio >= keep_frac) & (seen >= max(3, int(0.3 * len(views))))
    log(f"  카빙 {res}^3 · 남은 복셀 {int(occ.sum()):,} ({100*occ.mean():.2f}%)")
    return occ.reshape(res, res, res), ax


def bbox_from_masks(views, masks, seed_pts, pad=0.15, keep_frac=0.9, log=print):
    """마스크 안에 꾸준히 들어오는 점만 남겨 물체 상자를 잡는다.

    seed_pts 는 COLMAP sparse 점군. 배경까지 들어 있으므로 여기서 걸러낸다.
    """
    inside, seen = _count_inside(np.asarray(seed_pts, np.float32), views, masks)
    r = np.where(seen > 0, inside / np.maximum(seen, 1), 0.0)
    sel = r >= keep_frac
    if sel.sum() < 30:
        sel = r >= np.percentile(r, 99.0)
        log(f"  [!] 상자 잡기: 기준을 낮춰 상위 1% 만 씁니다 ({int(sel.sum())}점)")
    Q = np.asarray(seed_pts)[sel]
    lo, hi = Q.min(0), Q.max(0)
    span = hi - lo
    lo = lo - span * pad
    hi = hi + span * pad
    log(f"  물체 상자 {np.round(lo,3)} ~ {np.round(hi,3)} "
        f"(씨앗 {int(sel.sum()):,}/{len(seed_pts):,}점)")
    return lo, hi


def to_mesh(occ, ax, smooth=8, log=print):
    """occupancy → marching cubes → 메쉬(open3d)."""
    from skimage import measure
    import open3d as o3d

    # 경계에 벽을 세워 바깥과 이어지지 않게 한다. 없으면 상자 면이 열린다.
    pad = np.pad(occ.astype(np.float32), 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = measure.marching_cubes(pad, level=0.5)
    verts -= 1.0                                    # 패딩 보정
    step = np.array([a[1] - a[0] for a in ax], np.float64)
    origin = np.array([a[0] for a in ax], np.float64)
    V = origin + verts * step

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(faces)
    m.remove_degenerate_triangles()
    m.remove_duplicated_vertices()
    lab = np.asarray(m.cluster_connected_triangles()[0])
    if len(lab):
        m.remove_triangles_by_mask(lab != np.bincount(lab).argmax())
        m.remove_unreferenced_vertices()
    if smooth > 0:
        m = m.filter_smooth_taubin(number_of_iterations=int(smooth))
    m.compute_vertex_normals()
    log(f"  메쉬 정점 {len(m.vertices):,} · 삼각형 {len(m.triangles):,}")
    return m
