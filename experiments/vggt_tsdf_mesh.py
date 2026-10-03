import os
"""VGGT → 깊이맵 → TSDF 융합 → 메쉬 → STL. COLMAP 을 안 쓴다.

    python vggt_tsdf_mesh.py <사진폴더> --sam "stepper motor|the object in the center"

왜 이 경로인가 — COLMAP 덴스 MVS 는 patch_match 에만 40~60분이 든다. VGGT 는
깊이맵을 한 번의 순전파로 내놓으므로 같은 일이 분 단위로 끝난다. 정확도는
COLMAP 이 낫지만(실측 평면 잔차 1.37% vs 5.13%), 형상 확인과 3D 프린트에는
이쪽으로 충분한지 재볼 가치가 있다.

왜 점군이 아니라 TSDF 인가 — 프레임마다 추정 깊이가 조금씩 달라서 점으로
합치면 같은 면이 두툼한 솜뭉치가 되고, Poisson 이 그 두께를 형상으로 오해해
표면을 우둘투둘하게 깎는다. TSDF 는 같은 복셀에 들어온 값을 평균내므로
프레임이 많을수록 표면이 오히려 매끄러워진다.

⚠ 치수는 여전히 못 믿는다. VGGT 깊이맵에는 중심에서 가장자리로 휘는 접시
  왜곡이 있고, TSDF 는 프레임 간 불일치를 고칠 뿐 그 체계적 왜곡은 못 고친다.
"""
import argparse, glob, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# experiments/ 에서 실행해도 상위 폴더의 server·colmap_pipe 를 찾게 한다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(folder, sam_prompt, out_dir, res=518, conf_pct=90.0,
        voxel_div=384, smooth=10, size_mm=0.0, log=print):
    import numpy as np
    import open3d as o3d
    import server                      # import 시점에 VGGT 를 GPU 로 올린다

    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if len(paths) < 3:
        raise RuntimeError(f"사진이 {len(paths)}장뿐입니다")
    log(f"사진 {len(paths)}장 · 해상도 {res}")

    masks = None
    if sam_prompt:
        t = time.time()
        masks = server.run_sam(paths, sam_prompt, 1024, log)
        got = sum(1 for m in (masks or []) if m is not None)
        log(f"SAM {got}/{len(paths)}장 · {time.time()-t:.0f}s")
        if not got:
            raise RuntimeError("마스크를 하나도 못 얻었습니다 — 프롬프트를 바꾸세요")

    t = time.time()
    mesh = server.reconstruct_tsdf(paths, res, masks, conf_pct,
                                   log=log, voxel_div=voxel_div)
    log(f"TSDF {time.time()-t:.0f}s")

    # TSDF 는 안 본 곳을 구멍으로 남기고 주변에 작은 파편을 흘린다.
    lab = np.asarray(mesh.cluster_connected_triangles()[0])
    if len(lab):
        n0 = len(np.asarray(mesh.triangles))
        mesh.remove_triangles_by_mask(lab != np.bincount(lab).argmax())
        mesh.remove_unreferenced_vertices()
        log(f"파편 제거 {n0:,} → {len(np.asarray(mesh.triangles)):,}")
    if smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(smooth))
        mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    ext = mesh.get_max_bound() - mesh.get_min_bound()
    if size_mm > 0:
        k = size_mm / float(ext.max())
        mesh.scale(k, center=mesh.get_center())
        ext = mesh.get_max_bound() - mesh.get_min_bound()
        log(f"스케일 앵커: 최대변 → {size_mm}mm (x{k:.4g})")

    ply = os.path.join(out_dir, "mesh_vggt_tsdf.ply")
    stl = os.path.join(out_dir, "mesh_vggt_tsdf.stl")
    o3d.io.write_triangle_mesh(ply, mesh)
    o3d.io.write_triangle_mesh(stl, mesh)
    log(f"삼각형 {len(np.asarray(mesh.triangles)):,} · 크기 {np.round(ext,2)}")
    log(f"→ {ply}")
    log(f"→ {stl}")
    return dict(ply=ply, stl=stl,
                n_triangles=int(len(np.asarray(mesh.triangles))),
                extent=[round(float(v), 3) for v in ext])


def main():
    ap = argparse.ArgumentParser(description="VGGT 깊이맵 → TSDF → 메쉬")
    ap.add_argument("folder")
    ap.add_argument("--sam", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--conf", type=float, default=90.0, help="신뢰도 상위 %%")
    ap.add_argument("--voxel-div", type=int, default=384,
                    help="복셀 = 물체크기/이 값. 크면 촘촘하고 느리다")
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--size-mm", type=float, default=0.0,
                    help="최대변의 실제 길이(mm). 치수 앵커")
    a = ap.parse_args()
    folder = os.path.abspath(os.path.expanduser(a.folder.strip().strip('"')))
    out = a.out or os.path.join(folder, "_tsdf")
    run(folder, a.sam, out, a.res, a.conf, a.voxel_div, a.smooth, a.size_mm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
