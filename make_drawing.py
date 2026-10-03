"""
3D 점군(mm 단위) → 기술 도면 PDF (3면도 + 치수) + STL
사용법:
  python3 make_drawing.py [입력ply] [물체명] [up축]
   예) python3 make_drawing.py ~/vggt/result_chain.ply "부품A" y
  * 입력 ply는 mm 단위로 스케일된 것이어야 함 (chain2.py에서 스케일 보정한 결과)
  * up축: 위쪽 방향 축 (x/y/z, 기본 y). 도면 방향이 이상하면 바꿔보세요.
"""
import os, sys, datetime
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from scipy import ndimage
from skimage import measure

# ---- 한글 폰트 ----
_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NotoSansKR.ttf")
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def ptp(a):
    return float(a.max() - a.min())


def silhouette(pts2d, grid_mm=None, ci=2):
    """점을 격자에 찍어 채운 실루엣의 윤곽선 (오목 형태 보존).
    grid_mm=None이면 물체 크기에 맞춰 자동 (약 200칸 해상도)."""
    if len(pts2d) < 10:
        return []
    span = max(pts2d[:, 0].max() - pts2d[:, 0].min(),
               pts2d[:, 1].max() - pts2d[:, 1].min())
    if grid_mm is None:
        grid_mm = max(span / 200.0, 1e-6)   # 물체 크기 무관하게 ~200칸
    mn = pts2d.min(0) - grid_mm * 3
    mx = pts2d.max(0) + grid_mm * 3
    nx = int((mx[0] - mn[0]) / grid_mm) + 1
    ny = int((mx[1] - mn[1]) / grid_mm) + 1
    if nx < 3 or ny < 3 or nx * ny > 6_000_000:
        return []
    g = np.zeros((ny, nx), bool)
    ix = np.clip(((pts2d[:, 0] - mn[0]) / grid_mm).astype(int), 0, nx - 1)
    iy = np.clip(((pts2d[:, 1] - mn[1]) / grid_mm).astype(int), 0, ny - 1)
    g[iy, ix] = True
    g = ndimage.binary_dilation(g, iterations=ci)
    g = ndimage.binary_fill_holes(g)
    g = ndimage.binary_erosion(g, iterations=ci)
    outs = []
    for c in measure.find_contours(g.astype(float), 0.5):
        outs.append(np.column_stack([c[:, 1] * grid_mm + mn[0], c[:, 0] * grid_mm + mn[1]]))
    return outs


def dim_line(ax, x0, y0, x1, y1, text, off=0):
    """치수선 (양끝 화살표 + 숫자)"""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(mx, my + off, text, ha="center", va="bottom",
            fontsize=8, color="#c0392b", fontweight="bold")


def draw_view(ax, pts, la, lb, title):
    for c in silhouette(pts):
        ax.plot(c[:, 0], c[:, 1], "k-", lw=1.6)
        ax.fill(c[:, 0], c[:, 1], color="#d6e4f0", alpha=0.6)
    w = ptp(pts[:, 0]); h = ptp(pts[:, 1])
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    m = max(w, h) * 0.12 + 1
    # 가로 치수(아래)
    dim_line(ax, x0, y0 - m, x1, y0 - m, f"{w:.0f}")
    # 세로 치수(왼쪽)
    dim_line(ax, x0 - m, y0, x0 - m, y1, f"{h:.0f}")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, ls=":")
    ax.set_xlabel(f"{la} (mm)", fontsize=8)
    ax.set_ylabel(f"{lb} (mm)", fontsize=8)
    ax.margins(0.18)


def make_drawing(ply_in, name="물체", up="y",
                 out_pdf=None, out_stl=None, make_stl=True):
    ply_in = os.path.expanduser(ply_in)
    m = trimesh.load(ply_in)
    P = np.asarray(m.vertices, dtype=float)
    if len(P) < 10:
        print("[!] 점이 너무 적음"); return
    print(f"점 {len(P)}개 불러옴")

    # up축이 Y(인덱스1)로 오도록 좌표 재배열.
    #
    # ⚠ 반드시 순환치환(cyclic)이어야 한다. 예전에는 up축만 빼고 나머지를 순서대로
    # 붙였는데(up='z' → [x, z, y]), 그 사상은 행렬식이 -1인 **반사**라 도면이
    # 좌우로 뒤집혀 나왔다. up='y'(기본)만 항등이라 멀쩡했고, rig.py 의 UP_AXIS 가
    # "z" 이므로 좌표(poses.json)로 찍은 스캔은 매번 거울상이 됐다.
    # 순환치환은 행렬식이 +1이라 손잡이(handedness)가 보존된다.
    CYCLE = {"x": (2, 0, 1),      # (z, x, y)
             "y": (0, 1, 2),      # (x, y, z)  항등
             "z": (1, 2, 0)}      # (y, z, x)
    a, b, c = CYCLE.get(up.lower(), CYCLE["y"])
    P2 = np.column_stack([P[:, a], P[:, b], P[:, c]])

    dims = np.array([ptp(P2[:, 0]), ptp(P2[:, 1]), ptp(P2[:, 2])])
    print(f"치수(mm): 폭 {dims[0]:.1f} · 높이 {dims[1]:.1f} · 깊이 {dims[2]:.1f}")

    # ===== 도면 PDF =====
    if out_pdf is None:
        out_pdf = os.path.expanduser("~/vggt/도면.pdf")
    fig = plt.figure(figsize=(16.5, 11.7))  # A3 가로 비율
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.28)

    ax_f = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 0])
    ax_iso = fig.add_subplot(gs[0, 2], projection="3d")
    ax_info = fig.add_subplot(gs[1, 1:])
    ax_info.axis("off")

    # 정면도(X-Y), 측면도(Z-Y), 평면도(X-Z)
    draw_view(ax_f, P2[:, [0, 1]], "X", "Y", "정면도 (Front)")
    draw_view(ax_s, P2[:, [2, 1]], "Z", "Y", "측면도 (Side)")
    draw_view(ax_t, P2[:, [0, 2]], "X", "Z", "평면도 (Top)")

    # 아이소메트릭 점군 미리보기
    sub = P2[np.random.default_rng(0).choice(len(P2), min(6000, len(P2)), replace=False)]
    ax_iso.scatter(sub[:, 0], sub[:, 2], sub[:, 1], s=1, c=sub[:, 1], cmap="viridis")
    ax_iso.set_title("아이소메트릭", fontsize=11, fontweight="bold")
    ax_iso.set_xlabel("X"); ax_iso.set_ylabel("Z"); ax_iso.set_zlabel("Y")
    try:
        ax_iso.set_box_aspect((dims[0], dims[2], dims[1]))
    except Exception:
        pass

    # 정보 표(도면틀)
    today = datetime.date.today().isoformat()
    diag = float(np.linalg.norm(dims))
    info = [
        ["항목", "값"],
        ["물체명", name],
        ["전체 폭 (X)", f"{dims[0]:.1f} mm"],
        ["전체 높이 (Y)", f"{dims[1]:.1f} mm"],
        ["전체 깊이 (Z)", f"{dims[2]:.1f} mm"],
        ["대각선", f"{diag:.1f} mm"],
        ["점 개수", f"{len(P):,}"],
        ["단위", "mm"],
        ["작성일", today],
        ["비고", "3D 스캔 기반 · 대략 치수"],
    ]
    tbl = ax_info.table(cellText=info, cellLoc="left", loc="center",
                        colWidths=[0.35, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#8a97a8")
        if r == 0:
            cell.set_facecolor("#1B2A4A")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F5F9")

    fig.suptitle(f"3D 스캔 도면 — {name}", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015,
             "본 도면은 3D 스캔 점군에서 자동 생성된 대략 치수 도면입니다. 정밀 제조용이 아닙니다.",
             ha="center", fontsize=8, color="#5A6472")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"도면 저장: {out_pdf}")

    # ===== STL (선택, 대략적) =====
    if not make_stl:
        print("STL 생성 건너뜀 (make_stl=False)")
        return out_pdf
    if out_stl is None:
        out_stl = os.path.expanduser("~/vggt/model.stl")
    try:
        wt = point_cloud_to_stl(P, out_stl)
        print(f"STL 저장: {out_stl}")
        if wt:
            print("  닫힌 입체(watertight) — 프린팅에 유리")
        else:
            print("  ※ 완전히 닫히지 않음. 슬라이서(Cura 등)의 자동수리로 대개 프린팅 가능.")
        print("  ※ 스캔 기반이라 표면이 매끈하지 않을 수 있음. 슬라이서에서 확인 권장.")
    except Exception as e:
        print(f"[STL 건너뜀] {e}")
        print("  STL은 open3d가 필요하고 메모리를 많이 씁니다. 노트북에서 돌리는 걸 권장.")

    return out_pdf


def point_cloud_to_stl(P, out_stl, depth=9, trim=0.03, voxel_frac=0.005,
                       smooth=15):
    """점군 → 메쉬(Poisson) → 구멍메우기 → STL. watertight 여부 반환.

    voxel_frac·depth 기본값 주의 — 예전 값(0.01 / depth 8)은 너무 거칠었다.
    26만 점짜리 점군에서 14,647점만 남기고 재구성해서 기어 이빨이 통째로 뭉개졌다.
    실측(기어 달린 모터, 261,092점):
        0.01  / d8  →  14,647점 →  80,106 삼각형   이빨 없음, 플레이트 모서리 뭉개짐
        0.005 / d9  →  73,898점 → 406,784 삼각형   이빨·나사구멍까지 보임  ← 기본값
        0.003 / d10 → 160,893점 → 498,658 삼각형   비슷하나 3배 느림
        다운샘플 없음/d10 → 261,092점 → 1,069,027   법선 반경이 작아져 오히려 뭉개짐
    voxel_frac=0 이면 다운샘플을 건너뛴다(권장하지 않음)."""
    import open3d as o3d
    P = np.asarray(P, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    vs = diag * (voxel_frac if voxel_frac > 0 else 0.002)
    if voxel_frac > 0:
        pcd = pcd.voxel_down_sample(vs)
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 5, max_nn=30))
    try:
        pcd.orient_normals_consistent_tangent_plane(15)
    except Exception:
        pass
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    # Poisson은 데이터 밖으로 부풀어 → 점군 바운딩박스로 자르기
    bb = pcd.get_axis_aligned_bounding_box()
    bb = bb.scale(1.05, bb.get_center())
    mesh = mesh.crop(bb)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices()
    lab = np.asarray(mesh.cluster_connected_triangles()[0])
    if len(lab) > 0:
        mesh.remove_triangles_by_mask(lab != np.bincount(lab).argmax())
        mesh.remove_unreferenced_vertices()
    # VGGT 점군은 프레임마다 깊이가 조금씩 달라서 표면이 '두툼한 솜뭉치'가 된다.
    # Poisson이 그 사이를 헤집고 지나가면 바위처럼 우둘투둘한 면이 나온다.
    # Taubin은 부피를 보존하며 고주파 잡음만 깎아서, 기어 이빨 같은 큰 특징은
    # 남기고 표면만 정리한다. 15~40이 적당했고 그 이상은 더 나아지지 않았다.
    if smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth))
        mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    mt = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    mt.remove_unreferenced_vertices()
    mt.fill_holes()
    mt.fix_normals()
    mt.export(out_stl)
    return bool(mt.is_watertight)


if __name__ == "__main__":
    ply = sys.argv[1] if len(sys.argv) > 1 else "~/vggt/result_chain.ply"
    name = sys.argv[2] if len(sys.argv) > 2 else "물체"
    up = sys.argv[3] if len(sys.argv) > 3 else "y"
    out = make_drawing(ply, name=name, up=up)
    try:
        import subprocess
        subprocess.Popen(["xdg-open", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
