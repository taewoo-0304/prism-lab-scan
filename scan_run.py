"""사진 폴더 하나 → 점군·메쉬·STL. 묻는 것 없음.

    python scan_run.py <사진폴더>
    python scan_run.py <사진폴더> --json      # UI 연동용 (진행률을 JSON 한 줄씩)

왜 물어보지 않는가
    시료는 항상 한 개고 턴테이블 한가운데 있다. 그러면 사람이 정해줄 게 없다.
    SAM 프롬프트도, 필터 세기도 데이터에서 정할 수 있다.

    예전에는 SAM 프롬프트를 사람이 쳤는데 그게 품질을 좌우했다(같은 사진에서
    물체 덮음률 45.2% ~ 91.6%). 게다가 '|' 가 들어가면 배치 파일이 죽었다.
    여기서는 후보를 순서대로 시도하고, 마스크가 잡힌 프레임 수와 마스크 크기로
    쓸 만한지 판정한다.

품질을 위해 하는 일 (실측 0831_2058, 133장)
    서버가 SAM 마스크를 잡으면 거리 필터를 자동으로 풀어버려서(dist_pct=100)
    배경이 통째로 들어온다. 그 결과 중심거리 최대가 물체 크기의 17배였다.
    dist_pct 를 직접 줘도 멀리 뭉친 배경 덩어리는 살아남는다(최대 4.06).
    그래서 서버 결과를 받은 뒤 두 단계로 정리한다.
        군집   768,422 → 757,172점 · 대각선 7.502 → 1.602
        후광   757,172 → 361,944점 · 대각선 → 0.484 · 중심거리 최대 0.221
    예전에 '잘 나왔다'던 결과가 대각선 0.465 / 최대 0.244 였으니 같은 수준인데,
    그쪽은 턴테이블 판이었고 이쪽은 물체 본체다.
"""
import argparse, base64, glob, io, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HOST, PORT = "127.0.0.1", 5000

# 후보를 전부 돌려보고 '마스크를 가장 많이 확보한' 것을 고른다.
#
# 실측 2026-09-25, 6개 촬영 폴더 x 표본 16장 (마스크 확보 장수):
#                                              2058 2112 2119 2139 2044 2130
#   the object in the center                     13   11   11   13   15    8
#   + the small object + the small machine part  16   16   16   14   16   16
# 한 문구로는 물체가 바뀔 때마다 3~8장씩 놓친다. 세 개를 합집합으로 걸면
# 전 폴더에서 다 잡았다. 면적도 안 부푼다(중앙 6~16%).
#
# ⚠ 'the object on the turntable' 은 넣지 마라. 어떤 프레임에서 턴테이블을
#   통째로 삼켜 면적이 50.97% 까지 갔다. 'the object' / 'the device in the
#   middle' 는 반대로 한 장도 못 잡았다(0/16).
SAM_CANDIDATES = [
    "the object in the center|the small object|the small machine part",
    "the object in the center|the small machine part",
    "the object in the center",
]
# 쓸 만한 마스크의 조건. 면적은 '중앙값'으로 본다 — 최대값으로 보면 한 프레임만
# 튀어도 후보가 통째로 탈락한다(0831_2130 이 그렇다: 중앙 9.92% 인데 최대 62.8%).
MIN_FRAMES_FRAC = 0.70      # 이 비율만큼 프레임에서 마스크가 나와야 한다
MIN_AREA, MAX_AREA = 0.01, 0.35   # 프레임 면적 대비 중앙값


def emit(js, stage, state, pct=None, msg=None, **kw):
    """UI 연동용 한 줄 JSON. --json 이 아니면 사람이 읽는 줄로 찍는다."""
    if js:
        d = {"stage": stage, "state": state}
        if pct is not None:
            d["pct"] = round(float(pct), 1)
        if msg:
            d["msg"] = msg
        d.update(kw)
        print(json.dumps(d, ensure_ascii=False), flush=True)
    elif msg:
        print(f"  {msg}", flush=True)


PROBE_FRAMES = 12          # 프롬프트 고르기용 표본 수


def pick_prompt(paths, log=print):
    """프롬프트 후보 중 하나를 고른다. 마스크는 버리고 이름만 돌려준다.

    ⚠ 전체 프레임에 돌리지 마라. 본 마스킹은 서버가 어차피 다시 한다
      (server.py 가 sam_prompt 를 받아 스스로 SAM 을 돌린다). 여기서 전부
      돌리면 같은 일을 두 번 하는 셈이고, 133장이면 그것만 125초다.
      프롬프트가 맞는지는 골고루 뽑은 12장이면 충분히 갈린다 —
      좋은 프롬프트와 나쁜 프롬프트의 차이가 워낙 커서(덮음률 91.6% 대 45.2%)
      표본이 작아도 순위가 안 바뀐다.
    """
    import numpy as np
    import sam_mask

    ok, why = sam_mask.available()
    if not ok:
        log(f"SAM 사용 불가 ({why}) — 마스크 없이 진행합니다")
        return None

    idx = np.linspace(0, len(paths) - 1, min(PROBE_FRAMES, len(paths))).astype(int)
    probe = [paths[i] for i in sorted(set(idx.tolist()))]
    quiet = lambda *a, **k: None
    best = (None, -1.0)
    for prompt in SAM_CANDIDATES:
        t = time.time()
        try:
            masks = sam_mask.segment(probe, prompt=prompt, device="cuda", log=quiet)
        except Exception as e:
            log(f"SAM 실패 ({type(e).__name__}) — 다음 후보로")
            continue
        finally:
            try:
                sam_mask.unload()
            except Exception:
                pass
        got = [m for m in masks if m is not None]
        frac = len(got) / max(len(probe), 1)
        area = float(np.median([m.mean() for m in got])) if got else 0.0
        ok = MIN_AREA <= area <= MAX_AREA
        log(f"프롬프트 시험 '{prompt}' → {len(got)}/{len(probe)}장 · "
            f"물체 비율 중앙 {100*area:.1f}%{'' if ok else ' (면적 이상 — 제외)'} · "
            f"{time.time()-t:.0f}s")
        # 첫 통과에서 멈추지 않는다. 전에 그렇게 했다가 1순위가 기준만 겨우
        # 넘기는 바람에, 마스크를 더 많이 잡는 후보를 못 보고 지나쳤다.
        if ok and frac > best[1]:
            best = (prompt, frac)
        if best[1] >= 0.999:        # 전 프레임 확보. 더 볼 것 없다
            break
    if best[0] is not None and best[1] >= MIN_FRAMES_FRAC:
        log(f"프롬프트 선택: '{best[0]}' ({100*best[1]:.0f}% 프레임)")
        return best[0]
    if best[0] is not None:
        log(f"확보율이 낮지만 가장 나은 후보를 씁니다: '{best[0]}' "
            f"({100*best[1]:.0f}%)")
        return best[0]
    log("쓸 만한 마스크를 못 얻었습니다 — 마스크 없이 진행합니다")
    return None


def run(folder, out_dir=None, every=1, make_mesh=True, keep_pct=50.0,
        depth=9, smooth=15, precise=False, js=False, log=print):
    import numpy as np
    import requests
    import trimesh
    from object_clean import clean_object, remove_halo, denoise

    folder = os.path.abspath(folder)
    out_dir = out_dir or os.path.join(folder, "_scan")
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()
    res = {"folder": folder, "out_dir": out_dir}

    # ---- 1단계: 준비 확인 -------------------------------------------------
    emit(js, 1, "running", 0, "장비·서버 확인")
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    n_all = len(paths)
    if every > 1:
        paths = paths[::every]
    if len(paths) < 3:
        emit(js, 1, "error", msg=f"사진이 {len(paths)}장뿐입니다")
        raise RuntimeError(f"사진이 {len(paths)}장뿐입니다 (jpg, 하위폴더 제외)")
    try:
        h = requests.get(f"http://{HOST}:{PORT}/health", timeout=5).json()
    except Exception:
        emit(js, 1, "error", msg="VGGT 서버가 꺼져 있습니다")
        raise RuntimeError("VGGT 서버가 안 떠 있습니다. VGGT-Server.bat 을 먼저 실행하세요")
    if not h.get("alive"):
        raise RuntimeError("서버가 응답은 하는데 준비되지 않았습니다")
    log(f"사진 {len(paths)}장" + (f" (전체 {n_all}장 중 {every}장마다)" if every > 1 else "")
        + f" · 서버 {h.get('device')} · SAM {'가능' if h.get('sam') else '없음'}")
    emit(js, 1, "done", 100, f"사진 {len(paths)}장 · 서버 준비됨",
         images=len(paths), device=h.get("device"))

    # ---- 2단계: 3D 복원 ---------------------------------------------------
    emit(js, 2, "running", 0, "프롬프트 고르는 중")
    prompt = pick_prompt(paths, log=log)
    res["sam_prompt"] = prompt

    emit(js, 2, "running", 25, "VGGT 추론")
    data = {"res": h.get("max_res", 518), "floor": 0, "noise_std": 1.5,
            "max_points": 0}
    if prompt:
        # 서버가 SAM 을 한 번 더 돌린다. 여기서 고른 프롬프트를 그대로 넘긴다.
        data["sam"] = 1
        data["sam_prompt"] = prompt
    files = [("images", (os.path.basename(p), open(p, "rb"), "image/jpeg"))
             for p in paths]
    t0 = time.time()
    try:
        r = requests.post(f"http://{HOST}:{PORT}/reconstruct", files=files,
                          data={k: str(v) for k, v in data.items()},
                          timeout=(30, None))
    finally:
        for _, (_, f, _) in files:
            f.close()
    if r.status_code != 200:
        emit(js, 2, "error", msg=f"서버 오류 {r.status_code}")
        raise RuntimeError(f"서버 오류 {r.status_code}: {r.text[:300]}")
    z = np.load(io.BytesIO(base64.b64decode(r.json()["npz_b64"])))
    P, C = z["points"], z["colors"]
    log(f"VGGT {len(P):,}점 · {time.time()-t0:.0f}s")

    emit(js, 2, "running", 70, "물체만 남기는 중")
    P, C = clean_object(P, C, log=log)
    P, C = remove_halo(P, C, keep_pct=keep_pct, log=log)
    P, C = denoise(P, C, log=log)

    ply = os.path.join(out_dir, "points.ply")
    trimesh.PointCloud(vertices=P, colors=C).export(ply)
    res["ply"] = ply
    res["n_points"] = int(len(P))
    png = None
    try:
        from localtest import save_preview
        png = save_preview(P, C, os.path.join(out_dir, "preview.png"))
        res["preview"] = png
    except Exception as e:
        log(f"(미리보기 이미지 건너뜀: {e})")

    if make_mesh:
        emit(js, 2, "running", 85, "메쉬 만드는 중")
        from make_drawing import point_cloud_to_stl
        stl = os.path.join(out_dir, "mesh.stl")
        wt = point_cloud_to_stl(P, stl, depth=depth, smooth=smooth)
        mt = trimesh.load_mesh(stl)
        mesh_ply = os.path.join(out_dir, "mesh.ply")
        mt.export(mesh_ply)
        res.update(stl=stl, mesh_ply=mesh_ply,
                   n_triangles=int(len(mt.faces)), watertight=bool(wt))
        log(f"메쉬 삼각형 {len(mt.faces):,} · 밀폐 {'예' if wt else '아니오'}")

    if precise:
        # ---- 정밀 복원 (COLMAP) --------------------------------------------
        # 왜 VGGT 로 안 끝내는가 — VGGT 깊이맵에는 프레임마다 같은 방향으로 휘는
        # 왜곡이 있어서 점군이 안장처럼 굽는다. 실측(2026-09-24, 0831_2058):
        # 정리를 마쳐 뜬점·후광을 다 걷어내고 대각선이 0.465 까지 줄어든 뒤에도
        # 메쉬는 닫히지 않는 굽은 껍데기였다. 법선을 바깥으로 강제해도 같았다.
        # 이건 후처리로 못 편다 — 깊이맵을 다시 만드는 수밖에 없다.
        emit(js, 2, "running", 92, "정밀 복원 (COLMAP) — 수십 분")
        import colmap_pipe
        t0 = time.time()
        try:
            pr = colmap_pipe.run_all(folder, prompt or "", out_dir=out_dir,
                                     log=log)
            res["precise"] = {k: pr[k] for k in
                              ("object_ply", "mesh_ply", "mesh_stl") if k in pr}
            res["precise"]["report"] = pr.get("report")
            log(f"정밀 복원 {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"[정밀 복원 실패] {e}")
            res["precise_error"] = str(e)

    res["elapsed_sec"] = round(time.time() - t_all, 1)
    emit(js, 2, "done", 100, f"3D 복원 완료 ({res['elapsed_sec']:.0f}초)", **{
        k: res[k] for k in ("ply", "stl", "mesh_ply", "preview", "n_points",
                            "n_triangles") if k in res})

    # ---- 3단계: 분광기 (미구현) -------------------------------------------
    emit(js, 3, "skipped", msg="분광기 미연결 — 재질 분석 건너뜀")

    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(description="사진 폴더 → 점군·메쉬·STL (입력 없음)")
    ap.add_argument("folder")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", action="store_true", help="진행률을 JSON 한 줄씩")
    ap.add_argument("--every", type=int, default=1, help="N장마다 1장만 사용")
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--keep-pct", type=float, default=50.0,
                    help="후광 제거 후 남길 비율. 얇은 물체면 올려라")
    ap.add_argument("--precise", action="store_true",
                    help="VGGT 뒤에 COLMAP 정밀 복원까지 (수십 분 더). "
                         "치수를 재거나 출력할 거면 이쪽 결과를 써라")
    a = ap.parse_args()

    js = a.json
    log = (lambda *x: None) if js else print
    folder = os.path.abspath(os.path.expanduser(a.folder.strip().strip('"')))
    if not os.path.isdir(folder):
        emit(js, 1, "error", msg=f"폴더가 없습니다: {folder}")
        if not js:
            print(f"[!] 폴더가 없습니다: {folder}")
        return 1
    if not js:
        print("=" * 62)
        print(f"  {folder}")
        print("=" * 62)
    try:
        r = run(folder, a.out or None, max(1, a.every), not a.no_mesh,
                a.keep_pct, precise=a.precise, js=js, log=log)
    except Exception as e:
        if not js:
            print(f"\n[실패] {e}")
        return 1
    if not js:
        print("\n" + "=" * 62)
        print(f"  점군   {r['ply']}  ({r['n_points']:,}점)")
        if "stl" in r:
            print(f"  메쉬   {r['mesh_ply']}  (삼각형 {r['n_triangles']:,})")
            print(f"  STL    {r['stl']}")
        print(f"  소요   {r['elapsed_sec']:.0f}초")
        print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        sys.exit(1)
