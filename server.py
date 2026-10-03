"""
노트북 계산 서버 v2 — 젯슨은 촬영만, 3D 계산은 전부 이 서버에서.

v1 대비 바뀐 것
  * 해상도 제한 해제: 기본 518 (VGGT 학습 해상도 = 실질 화질 상한)
  * 장수 제한 해제: 몇 장이든 받음. VRAM에 한 번에 안 들어가면
                    자동으로 묶음 체이닝(umeyama 정합)으로 이어붙임
  * OOM 나면 묶음 크기 → 해상도 순으로 자동으로 낮춰서 재시도 (죽지 않음)
  * chain2가 하던 후처리를 서버가 흡수: 최대거리 필터 · 바닥평면 제거 ·
    노이즈 제거 · mm 스케일 보정까지 끝내서 반환

사용법:  run_server.bat 더블클릭
"""
import os, sys, json, time, io, gc, base64, tempfile, threading, shutil
from pathlib import Path

# torch import 전에 걸어야 먹는 것들
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["HF_HUB_OFFLINE"] = "1"

WORK = Path.home() / "vggt_server"
CFG = WORK / "config.json"
if not CFG.exists():
    print("[!] 세팅이 안 됐습니다. 먼저 실행하세요:  python setup_laptop.py")
    input("엔터를 누르면 종료합니다...")
    sys.exit(1)

cfg = json.loads(CFG.read_text(encoding="utf-8"))
VGGT_DIR = cfg["vggt_dir"]
MODEL_DIR = cfg["model_dir"]
HW = cfg["hardware"]

try:
    import torch
except ImportError:
    vp = cfg.get("venv_python", "")
    print("[!] 가상환경으로 실행해야 합니다:")
    print(f'    "{vp}" server.py')
    print(f'    또는 {WORK}\\run_server.bat 더블클릭')
    input("엔터를 누르면 종료합니다...")
    sys.exit(1)

sys.path.insert(0, VGGT_DIR)

import numpy as np
import torch.nn.functional as F
from flask import Flask, request, jsonify
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# ===== 상한 =====
MAX_RES = 518               # VGGT 학습 해상도. 이 위로 올려도 품질은 안 오르고 메모리만 폭증
RES_LADDER = [518, 448, 392, 336, 280, 224]   # OOM 시 내려갈 사다리
DEF_OVERLAP = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 62)
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    VRAM = props.total_memory / 1e9
    print(f"장치: GPU — {torch.cuda.get_device_name(0)} (VRAM {VRAM:.1f}GB)")
else:
    VRAM = 0
    print("!" * 62)
    print("[!!] CUDA를 못 씁니다 → CPU로 돕니다 (매우 느림).")
    print("     torch가 CPU판이거나 드라이버 문제입니다. setup_laptop.py를 다시 돌리세요.")
    print("!" * 62)
print(f"해상도 상한 {MAX_RES} · 사진 장수 제한 없음(자동 체이닝)")
print("=" * 62)

print("모델 로딩 중...")
t0 = time.time()
model = VGGT.from_pretrained(MODEL_DIR)
if DEVICE == "cuda" and VRAM < 10:
    # 8GB급: 본체는 fp16, 정밀도 민감한 헤드만 fp32로 되돌림
    model = model.half().to(DEVICE)
    model.depth_head = model.depth_head.float()
    for h in ["point_head", "track_head", "camera_head"]:
        if hasattr(model, h):
            setattr(model, h, getattr(model, h).float())
else:
    model = model.to(DEVICE)
model.eval()
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print(f"모델 준비 완료 ({time.time()-t0:.1f}s)")

app = Flask(__name__)
GPU_LOCK = threading.Lock()
BUSY = {"now": False}


# ---------------- 메모리 ----------------
def is_oom(e):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    return "out of memory" in str(e).lower()


def guess_group(res):
    """한 번에 GPU에 올릴 프레임 수 추정. 틀려도 OOM 나면 자동으로 줄어듦.
    총 VRAM이 아니라 '지금 실제로 비어 있는' VRAM을 봄 — 윈도우 바탕화면/브라우저가
    이미 1~2GB를 쓰고 있어서 총량 기준으로 잡으면 반드시 OOM 난다."""
    if DEVICE != "cuda":
        return 3
    try:
        free_gb = torch.cuda.mem_get_info()[0] / 1e9   # 모델 가중치는 이미 빠진 값
    except Exception:
        free_gb = max(VRAM - 3.2, 0.8)
    free = max(free_gb - 1.0, 0.6)                     # 작업 버퍼 여유
    # 16:9 사진 기준 518에서 프레임당 518x294 ≈ 152k픽셀 → 대략 0.23GB (경험값).
    # v1의 강제 리사이즈(518x700)를 없애서 픽셀이 2.4배 줄었기 때문에 이 값도 내려감.
    per_frame = 0.23 * (res / 518.0) ** 2
    return max(2, min(48, int(free / per_frame)))


# ---------------- 기하 유틸 (chain2에서 이식) ----------------
def quat_to_mat(q):
    i, j, k, r = q                                # XYZW (scalar-last), VGGT 규약
    two_s = 2.0 / (q * q).sum()
    return np.array([
        [1 - two_s * (j*j + k*k), two_s * (i*j - k*r),     two_s * (i*k + j*r)],
        [two_s * (i*j + k*r),     1 - two_s * (i*i + k*k), two_s * (j*k - i*r)],
        [two_s * (i*k - j*r),     two_s * (j*k + i*r),     1 - two_s * (i*i + j*j)],
    ])


def cam_centers_from_pose_enc(pe):
    # pe: (S,9) = [T(3), quat_xyzw(4), fov(2)]  →  카메라 위치 C = -R^T @ T
    return np.array([-quat_to_mat(pe[s, 3:7]).T @ pe[s, :3] for s in range(pe.shape[0])])


def umeyama(src, dst):
    mu_s = src.mean(0); mu_d = dst.mean(0)
    s_c = src - mu_s; d_c = dst - mu_d
    H = s_c.T @ d_c / len(src)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d]); R = Vt.T @ D @ U.T
    var = (s_c ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ D) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_windows(prev_wp, prev_cf, wp, cf, overlap, max_frames=4):
    """겹침 프레임의 고신뢰 점끼리 맞춰서 상대 유사변환 산출 (2단계 residual trimming)

    overlap 은 build_groups 가 계산한 **실제** 겹침이다. 겹침이 크면 정합에 쓰는
    점이 과하게 늘어나므로 뒤쪽 max_frames 장만 쓴다. 이때도 짝은 정확히 맞춘다 —
    직전 묶음의 마지막 k장 ↔ 새 묶음의 [overlap-k, overlap) 번째 장.
    """
    k = max(1, min(overlap, max_frames))
    sp = []; dp = []
    for o in range(k):
        pa = prev_wp[-(k - o)].reshape(-1, 3); ca = prev_cf[-(k - o)].reshape(-1)
        j = overlap - k + o
        pb = wp[j].reshape(-1, 3); cb = cf[j].reshape(-1)
        good = (ca > np.percentile(ca, 80)) & (cb > np.percentile(cb, 80))
        dp.append(pa[good]); sp.append(pb[good])
    sp = np.concatenate(sp); dp = np.concatenate(dp)
    if len(sp) < 10:
        raise RuntimeError("겹침 구간에 쓸 만한 점이 너무 적습니다 (사진이 너무 튀거나 겹침 부족)")
    s, R, t = umeyama(sp, dp)
    for pct in (70, 80):
        spt = (s * (R @ sp.T)).T + t
        resid = np.linalg.norm(spt - dp, axis=1)
        keep = resid < np.percentile(resid, pct)
        if keep.sum() < 10:
            break
        sp, dp = sp[keep], dp[keep]
        s, R, t = umeyama(sp, dp)
    return s, R, t


def find_plane(S, thresh, n_iter=400, seed=0):
    rng = np.random.default_rng(seed)
    best = None; bestc = 0
    for _ in range(n_iter):
        idx = rng.choice(len(S), 3, replace=False)
        p1, p2, p3 = S[idx]
        n = np.cross(p2 - p1, p3 - p1); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn; d = -n @ p1
        c = (np.abs(S @ n + d) < thresh).sum()
        if c > bestc:
            bestc = c; best = (n, d)
    return best, bestc


def remove_ground_ransac(P, thickness=15, max_planes=5, min_inlier_ratio=0.06,
                         parallel_thresh=0.9, search_sub=20000, seed=0,
                         both_side_max=0.15, min_keep_ratio=0.02, min_keep_pts=5000,
                         log=print):
    """가장 큰 평면(바닥)과 그에 평행한 평면들을 제거. 안전장치 2개 포함.

    가드가 '얼마나 지우나'가 아니라 '얼마나 남나'를 본다는 점이 중요하다.
    예전에는 max_remove_ratio=0.9 로 "전체의 90%를 넘겨 지우면 중단"이었는데,
    책상 위 물체를 찍으면 배경이 실제로 90%를 넘는 게 정상이라 이 가드가
    매번 걸려서 바닥 제거가 통째로 무력화됐다.
    실측(32장, 이 저장소 데이터): 최대 평면 inlier 90.8% → 두께 0/15/40 어느 값을
    줘도 제거된 점이 0개로 완전히 동일했다.
    지금은 '지운 뒤 남는 점'이 물체라고 부를 수 있는 양(기본 2% 또는 5000점)
    아래로 떨어질 때만 막는다."""
    rng = np.random.default_rng(seed)
    N = len(P)
    if N < 200:
        return np.ones(N, bool)
    # 멀리 있는 배경 몇 점 때문에 장면이 부풀면 슬래브 두께가 커져서 작은 물체가
    # 통째로 지워진다. 2~98 퍼센타일로 재서 막는다.
    lo = np.percentile(P, 2, axis=0); hi = np.percentile(P, 98, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag < 1e-9:
        return np.ones(N, bool)
    thresh = diag * (thickness / 1000.0)
    keep = np.ones(N, bool); ref_n = None
    for it in range(max_planes):
        ia = np.where(keep)[0]
        if len(ia) < 200:
            break
        Pa = P[ia]
        S = Pa[rng.choice(len(Pa), search_sub, replace=False)] if len(Pa) > search_sub else Pa
        best, bestc = find_plane(S, thresh, seed=seed + it)
        if best is None or bestc < len(S) * min_inlier_ratio:
            break
        n, d = best
        sgn = Pa @ n + d
        far = np.abs(sgn) >= thresh
        if far.sum() > 0:
            pos = float((sgn[far] > 0).mean())
            both = min(pos, 1 - pos)
            if both > both_side_max:
                log(f"    평면{it+1}: 양쪽에 점 {both*100:.0f}% → 물체 관통면으로 판단, 제거 안 함")
                break
        if ref_n is None:
            ref_n = n
        elif abs(n @ ref_n) < parallel_thresh:
            break
        on = np.abs(Pa @ n + d) < thresh
        left = int(keep.sum() - on.sum())
        floor_pts = max(min_keep_pts, int(N * min_keep_ratio))
        if left < floor_pts:
            log(f"    평면{it+1}: 지우면 {left}개만 남음 (하한 {floor_pts}) "
                f"→ 중단 (물체 보호)")
            break
        keep[ia[on]] = False
        log(f"    평면{it+1}: {int(on.sum())}개 제거 → 남은 점 {left}")
    return keep


# ---------------- 추론 ----------------
_SHAPE_LOGGED = {"done": False}


def prep_images(paths, res):
    """load_and_preprocess_images가 이미 긴 변 518 · 14의 배수로 맞춰서 준다.
    (1280x720 사진 → 518x294)

    v1은 여기에 size=(res, res*1.357)을 강제로 씌웠는데, 이건 가로세로 비율을
    깨뜨린다: 518x294(비율 1.76) → 700x518(비율 1.35). 세로로 1.3배 늘어난
    그림이 들어가니 복원된 물체도 세로로 늘어나고, 도면 mm 치수가 그만큼 틀렸다.
    여기서는 비율을 유지한 채로 줄이기만 한다 (res=518이면 손 안 댐)."""
    images = load_and_preprocess_images(paths)
    if res < MAX_RES:
        h, w = images.shape[-2:]
        sc = res / float(MAX_RES)
        H = max(14, int(round(h * sc / 14)) * 14)
        W = max(14, int(round(w * sc / 14)) * 14)
        images = F.interpolate(images, size=(H, W), mode="bilinear", align_corners=False)
    return images


def infer_window(paths, res):
    images = prep_images(paths, res)
    if not _SHAPE_LOGGED["done"]:
        print(f"  입력 텐서 {tuple(images.shape)} (프레임당 "
              f"{images.shape[-2]*images.shape[-1]/1000:.0f}k 픽셀)", flush=True)
        _SHAPE_LOGGED["done"] = True
    images = images.to(DEVICE)
    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(images)
        else:
            pred = model(images)
    wp = pred["world_points"][0].float().cpu().numpy()
    cf = pred["world_points_conf"][0].float().cpu().numpy()
    df = pred["depth"][0].float().cpu().numpy()
    im = pred["images"][0].permute(0, 2, 3, 1).float().cpu().numpy()
    pe = pred["pose_enc"][0].float().cpu().numpy()
    del pred, images
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return wp, cf, df, im, pe


def build_groups(n, group, overlap):
    """묶음 목록과 '직전 묶음과 실제로 겹치는 프레임 수'를 함께 돌려준다.

    ⚠ 실제 겹침을 같이 주는 게 핵심이다. 마지막 묶음은 끝을 n에 맞추려고 뒤로
    당겨 붙이는데, 그러면 직전 묶음과의 겹침이 overlap 보다 커진다. 예전에는
    그 사실을 무시하고 늘 overlap 장이 겹친다고 가정해서 align_windows 가
    **엉뚱한 프레임끼리** 정합했고, 프레임이 중복으로 담겨 cams 개수가 사진 수와
    안 맞았다(그러면 fit_to_poses 가 조용히 건너뛰어 mm 스케일이 무효가 된다).

    묶음8·겹침2 실측: n=20 은 정상이지만 n=21 은 겹침 7·프레임 26장,
    n=30 은 겹침 4·32장, n=40 은 겹침 6·44장이 나왔다.
    """
    group = max(1, min(group, n))
    step = max(1, group - overlap)
    starts = []
    i = 0
    while i + group <= n:
        starts.append(i); i += step
    if not starts:
        starts = [0]
    if starts[-1] + group < n:
        starts.append(n - group)          # 끝을 맞추려고 당겨 붙임 → 겹침이 커진다
    out = []
    prev_end = None
    for s0 in starts:
        ov = 0 if prev_end is None else max(0, prev_end - s0)
        out.append((list(range(s0, s0 + group)), ov))
        prev_end = s0 + group
    return out


def reconstruct_chained(paths, res, group, overlap, dist_pct, conf_pct,
                        masks=None, log=print):
    """묶음 단위로 추론 → umeyama로 누적 정합. 한 묶음이면 그냥 단일 추론.
    masks: 전역 인덱스별 bool 마스크(원본 사진 크기) 또는 None."""
    n = len(paths)
    group = max(2, min(group, n))
    overlap = min(overlap, group - 1) if group > 1 else 0
    groups = build_groups(n, group, overlap)
    if len(groups) > 1 and min(ov for _, ov in groups[1:]) < 1:
        raise RuntimeError("겹침이 0인 묶음이 있습니다 — overlap 을 1 이상으로 주세요")
    log(f"  {n}장 → 묶음 {len(groups)}개 (묶음{group} 겹침{overlap} 해상도{res})")

    all_P = []; all_C = []; all_D = []; all_cams = []
    prev_wp = prev_cf = None
    cum_s, cum_R, cum_t = 1.0, np.eye(3), np.zeros(3)

    for gi, (g, ov) in enumerate(groups):
        wp, cf, df, im, pe = infer_window([paths[k] for k in g], res)
        if gi == 0:
            s_rel, R_rel, t_rel = 1.0, np.eye(3), np.zeros(3)
        else:
            s_rel, R_rel, t_rel = align_windows(prev_wp, prev_cf, wp, cf, ov)

        def to_base(pts):
            p = (s_rel * (R_rel @ pts.T)).T + t_rel
            return (cum_s * (cum_R @ p.T)).T + cum_t

        cams_base = to_base(cam_centers_from_pose_enc(pe))
        frames = range(len(g)) if gi == 0 else range(ov, len(g))   # 실제 겹침만큼 건너뜀
        for fidx in frames:
            all_cams.append(cams_base[fidx])
            P = wp[fidx].reshape(-1, 3); C = im[fidx].reshape(-1, 3)
            D = df[fidx].reshape(-1); CF = cf[fidx].reshape(-1)
            m = (D <= np.percentile(D, dist_pct)) & (CF >= np.percentile(CF, 100 - conf_pct))
            if masks is not None:
                mk = masks[g[fidx]]
                if mk is not None:
                    import sam_mask
                    m &= sam_mask.resize_mask(mk, wp.shape[1], wp.shape[2]).reshape(-1)
            P, C, D = P[m], C[m], D[m]
            all_P.append(to_base(P).astype(np.float32))
            # 색은 바로 uint8로 (RAM 16GB라 float로 들고 있으면 부담)
            all_C.append(np.clip(C * 255, 0, 255).astype(np.uint8))
            all_D.append((D * (cum_s * s_rel)).astype(np.float32))

        cum_s2 = cum_s * s_rel
        cum_R2 = cum_R @ R_rel
        cum_t2 = (cum_s * (cum_R @ t_rel)) + cum_t
        cum_s, cum_R, cum_t = cum_s2, cum_R2, cum_t2
        prev_wp, prev_cf = wp, cf
        log(f"    [{gi+1}/{len(groups)}] 누적 점 {sum(len(p) for p in all_P)}")

    return (np.concatenate(all_P), np.concatenate(all_C),
            np.concatenate(all_D), np.array(all_cams))


def reconstruct_adaptive(paths, res, group, overlap, dist_pct, conf_pct,
                         masks=None, log=print):
    """OOM 나면 묶음 크기 → 해상도 순으로 낮춰서 재시도. 절대 그냥 죽지 않음."""
    ladder = [res] + [r for r in RES_LADDER if r < res]
    for r in ladder:
        g = group if r == res else guess_group(r)
        first = True
        while g >= 2:
            try:
                return reconstruct_chained(paths, r, g, overlap, dist_pct,
                                           conf_pct, masks, log), r, g
            except Exception as e:
                if not is_oom(e):
                    raise
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                # 첫 실패에서는 추정치로 바로 점프하고, 그 뒤부터 1씩 줄인다
                if first:
                    first = False
                    nxt = min(g - 1, guess_group(r))
                else:
                    nxt = g - 1
                if nxt >= 2:
                    log(f"    [메모리 부족] 묶음 {g} → {nxt} 로 줄여서 재시도")
                g = nxt
        log(f"  [메모리 부족] 해상도 {r} 로는 안 됨 → 한 단계 낮춤")
    raise RuntimeError("메모리가 부족해서 가장 낮은 설정으로도 실패했습니다")


# ---------------- TSDF 융합 ----------------
def reconstruct_tsdf(paths, res, masks, conf_pct, log=print, voxel_div=384,
                     trunc_mult=4.0):
    """깊이맵을 점으로 쏟아붓지 않고, 부호거리 격자에 프레임별로 누적 평균한다.

    왜 이게 필요한가 — 프레임마다 추정 깊이가 조금씩 달라서, 점으로 합치면
    같은 면이 '두툼한 솜뭉치'가 된다. Poisson은 그 두께를 진짜 형상으로 보고
    표면을 우둘투둘하게 깎는다(스무딩으로 안 없어지는 이유). TSDF는 같은 복셀에
    들어온 값을 평균내므로 프레임이 많을수록 표면이 오히려 매끄러워진다.

    한계 — 모든 프레임이 같은 방향으로 휜 깊이를 주면(방사형 왜곡) 평균을 내도
    휜 채로 남는다. TSDF가 고치는 것은 프레임 간 불일치이지 체계적 왜곡이 아니다.
    안 본 영역은 지어내지 않고 구멍으로 남긴다 — 치수 도면에는 이쪽이 안전하다.
    """
    import open3d as o3d
    import torch as _t
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    wp, cf, df, im, pe = infer_window(paths, res)
    D = df[..., 0] if df.ndim == 4 else df          # (S,H,W)
    S, H, W = D.shape
    extr, intr = pose_encoding_to_extri_intri(
        _t.from_numpy(pe)[None].float(), (H, W))
    extr = extr[0].numpy(); intr = intr[0].numpy()

    # 유효 픽셀: 신뢰도 상위 conf_pct% + (있으면) SAM 마스크
    keep = np.ones_like(D, bool)
    if conf_pct < 100:
        for s_ in range(S):
            keep[s_] &= cf[s_] >= np.percentile(cf[s_], 100 - conf_pct)
    if masks is not None:
        import sam_mask
        for s_ in range(S):
            mk = masks[s_]
            if mk is not None:
                keep[s_] &= sam_mask.resize_mask(mk, H, W)

    if keep.sum() < 100:
        raise RuntimeError("TSDF에 쓸 유효 픽셀이 너무 적습니다")

    # 복셀 크기는 '남길 물체'의 크기에서 정한다 (장면 전체가 아니라)
    obj = wp[keep].reshape(-1, 3)
    ext = float(np.linalg.norm(obj.max(0) - obj.min(0)))
    voxel = ext / voxel_div
    trunc = voxel * trunc_mult
    dmax = float(np.percentile(D[keep], 99.5) * 1.5)
    log(f"  [TSDF] 물체크기 {ext:.4f} · 복셀 {voxel:.5f} (1/{voxel_div}) · 절단 {trunc:.5f}")

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    used = 0
    for s_ in range(S):
        d = D[s_].astype(np.float32).copy()
        d[~keep[s_]] = 0.0                      # 0 = 무효 (TSDF가 무시)
        if not np.any(d > 0):
            continue
        c = np.ascontiguousarray(
            np.clip(im[s_] * 255, 0, 255).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(c), o3d.geometry.Image(np.ascontiguousarray(d)),
            depth_scale=1.0, depth_trunc=dmax, convert_rgb_to_intensity=False)
        K = intr[s_]
        cam = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        E = np.eye(4); E[:3, :4] = extr[s_]      # world→cam (OpenCV 규약)
        vol.integrate(rgbd, cam, E)
        used += 1

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    log(f"  [TSDF] {used}/{S} 프레임 융합 → 정점 {len(mesh.vertices):,} "
        f"삼각형 {len(mesh.triangles):,}")
    return mesh


# ---------------- 후처리 ----------------
def calc_scale(P, cams, ref_mm, ref_mode, use_cam_center=False, log=None):
    if ref_mm <= 0 or len(P) < 10:
        return None
    if ref_mode == "d":
        if len(cams) < 1:
            return None
        centroid = cams.mean(0) if use_cam_center else np.median(P, axis=0)
        orbit = np.median(np.linalg.norm(cams - centroid, axis=1))
        if orbit <= 1e-9:
            return None
        s = ref_mm / orbit
        if log:
            log(f"  [스케일] 궤도반지름 {orbit:.4f}단위 = {ref_mm}mm → 배율 {s:.2f}")
        return s
    biggest = (P.max(0) - P.min(0)).max()
    if biggest <= 1e-9:
        return None
    s = ref_mm / biggest
    if log:
        log(f"  [스케일] 최대치수 {biggest:.4f}단위 = {ref_mm}mm → 배율 {s:.2f}")
    return s


def fit_to_poses(cams, poses_xyz, log=print):
    """VGGT가 추정한 카메라 중심 → 리그가 실제로 명령한 좌표(mm)로 유사변환 정합.

    얻는 것 세 가지:
      1) mm 배율이 '추정'이 아니라 '확정'  (타이핑한 카메라거리 불필요)
      2) 점군이 리그 좌표계로 정렬 → 도면 up축을 매번 고를 필요 없음
      3) 잔차 = 이 스캔이 얼마나 믿을 만한지. 지금까지 없던 품질 지표.
    반환: (s, R, t, rms_mm, max_mm) 또는 None"""
    if poses_xyz is None or len(cams) < 3 or len(poses_xyz) != len(cams):
        if poses_xyz is not None:
            log(f"  [!] 좌표 {len(poses_xyz)}개 ≠ 사진 {len(cams)}장 → 포즈 정합 건너뜀")
        return None
    s, R, t = umeyama(np.asarray(cams, float), np.asarray(poses_xyz, float))
    fitted = (s * (R @ np.asarray(cams, float).T)).T + t
    resid = np.linalg.norm(fitted - poses_xyz, axis=1)
    rms = float(np.sqrt((resid ** 2).mean())); mx = float(resid.max())
    orbit = float(np.median(np.linalg.norm(
        poses_xyz - np.asarray(poses_xyz).mean(0), axis=1)))
    log(f"  [포즈 정합] 배율 {s:.2f} · 잔차 RMS {rms:.1f}mm / 최대 {mx:.1f}mm")
    if orbit > 1e-6 and rms > orbit * 0.05:
        log(f"  [!] 잔차가 궤도반지름({orbit:.0f}mm)의 {rms/orbit*100:.0f}%입니다. "
            f"복원 궤적이 실제 궤적과 안 맞습니다 — 이 스캔은 치수를 믿지 마세요.")
    return s, R, t, rms, mx


def postprocess(P, C, D, cams, ref_mm, ref_mode, max_dist_mm, floor, noise_std,
                max_points=800_000, poses_xyz=None, log=print):
    # 0) 리그 좌표가 있으면 그걸로 스케일 확정 (필터보다 먼저 — 이후 전부가 mm 기준)
    fit = fit_to_poses(cams, poses_xyz, log)

    # 1) 절대 거리 필터 (배경 잘라내기) — 바닥/노이즈보다 먼저
    if max_dist_mm > 0:
        if fit is not None:
            s0 = fit[0]
        else:
            s0 = calc_scale(P, cams, ref_mm, "d", use_cam_center=True) if ref_mode == "d" else None
        if s0 is None:
            log("  [!] 최대거리 필터는 카메라거리 기준(d 모드)이 필요합니다 → 건너뜀")
        else:
            keep = (D * s0) <= max_dist_mm
            if keep.sum() < 100:
                log(f"  [!] 최대거리 {max_dist_mm/10:.0f}cm 로 자르면 점이 {int(keep.sum())}개뿐 → 건너뜀")
            else:
                log(f"  [최대거리 {max_dist_mm/10:.0f}cm] {len(P)} → {int(keep.sum())}")
                P, C, D = P[keep], C[keep], D[keep]

    # 2) 바닥 평면
    if floor > 0:
        keep = remove_ground_ransac(P, thickness=floor, log=log)
        P, C, D = P[keep], C[keep], D[keep]
        log(f"  [바닥평면 제거] → {len(P)}")

    # 3) 점 수 상한 — 100장이면 필터 후에도 3M점이 나오는데, kNN 배열이 그 자리에서
    #    800MB를 먹는다. 도면(실루엣 윤곽)에는 80만점이면 차고 넘친다.
    if 0 < max_points < len(P):
        idx = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        idx.sort()
        P, C, D = P[idx], C[idx], D[idx]
        log(f"  [점 수 상한] {max_points}개로 균일 샘플링 → {len(P)}")

    # 4) 노이즈 (kNN 평균거리)
    if noise_std > 0 and len(P) > 100:
        from sklearn.neighbors import NearestNeighbors
        t0 = time.time()
        # n_jobs 기본값은 1코어다. 7940HX 16코어를 다 쓰게 해야 100장에서 안 답답하다
        nn = NearestNeighbors(n_neighbors=17, n_jobs=-1).fit(P)
        dists, _ = nn.kneighbors(P)
        md = dists[:, 1:].mean(axis=1)
        keep = md < md.mean() + noise_std * md.std()
        P, C, D = P[keep], C[keep], D[keep]
        del dists
        log(f"  [노이즈] → {len(P)}  ({time.time()-t0:.1f}s)")

    # 5) 스케일 확정
    info = {"source": "none", "rms_mm": None, "max_mm": None}
    dims = None
    if fit is not None:
        # 리그 좌표계로 회전+평행이동까지 적용 → 도면 up축이 매번 같아짐
        s, R, t, rms, mx = fit
        P = (s * (R @ P.T)).T + t
        cams = (s * (R @ np.asarray(cams, float).T)).T + t
        scale = s
        info = {"source": "poses", "rms_mm": round(rms, 2), "max_mm": round(mx, 2)}
    else:
        scale = calc_scale(P, cams, ref_mm, ref_mode, log=log)
        if scale is not None:
            P = P * scale
            cams = cams * scale
            info["source"] = "orbit" if ref_mode == "d" else "size"
    if scale is not None:
        ext = P.max(0) - P.min(0)
        dims = [float(ext[0]), float(ext[1]), float(ext[2])]
        log(f"  ===== 실측 치수 (mm) =====")
        log(f"    X {ext[0]:.1f} · Y {ext[1]:.1f} · Z {ext[2]:.1f}")
    return P, C, cams, scale, dims, info


# ---------------- SAM 3 ----------------
try:
    import sam_mask
    SAM_OK, SAM_WHY = sam_mask.available()
except Exception as e:
    sam_mask = None
    SAM_OK, SAM_WHY = False, f"sam_mask.py 를 못 읽음 ({type(e).__name__})"
print(f"SAM 3: {'사용 가능' if SAM_OK else '없음 — ' + SAM_WHY}")


def run_sam(paths, prompt, sam_size, log):
    """마스크만 만들고 곧바로 VRAM을 돌려준다.
    8GB에 VGGT(약 3.2GB)가 이미 올라가 있으니, 부족하면 VGGT를 잠깐 CPU로 내린다."""
    global model
    if not SAM_OK:
        log(f"  [SAM 건너뜀] {SAM_WHY}")
        return None

    from PIL import Image
    w, h = Image.open(paths[0]).size
    if h > w:
        log("  [!] 세로가 더 긴 사진입니다. VGGT 전처리가 세로를 잘라내는 구간이라 "
            "마스크가 어긋날 수 있습니다 (가로로 촬영 권장)")

    t0 = time.time()
    moved = False
    try:
        # VGGT를 먼저 CPU로 내린다. SAM 도는 동안 VGGT는 아무것도 안 하는데
        # 8GB 중 3.2GB를 잡고 있어서, 남은 자리에 SAM이 간신히 들어간다.
        #
        # 왜 OOM 예외를 기다리면 안 되는가 — 윈도우(WDDM)는 VRAM이 모자라도
        # OOM을 내지 않고 시스템 RAM으로 흘려보낸다. 그래서 예외는 영영 안 오고
        # 대신 조용히 느려진다. 실측(2026-09-05, RTX 3070 8GB, 사진 133장):
        #   VGGT를 GPU에 둔 채  →  프레임당 약 30초, 121장에서 사실상 정지
        #                          (vram_free 0.0GB, 클라이언트 1시간 타임아웃)
        # 모델 왕복 이동은 5~10초다. 프레임당 30초에 비하면 없는 값이다.
        if DEVICE == "cuda":
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            log(f"  [SAM 준비] VGGT를 CPU로 내립니다 (여유 VRAM {free_gb:.1f}GB)")
            model = model.to("cpu"); moved = True
            torch.cuda.empty_cache(); gc.collect()
        try:
            return sam_mask.segment(paths, prompt, DEVICE, sam_size, log=log)
        except Exception as e:
            if not is_oom(e) or DEVICE != "cuda":
                raise
            log("  [SAM 메모리 부족] CPU로 재시도")
            sam_mask.unload()
            torch.cuda.empty_cache(); gc.collect()
            return sam_mask.segment(paths, prompt, "cpu", sam_size, log=log)
    except Exception as e:
        log(f"  [SAM 실패] {e} → 마스크 없이 진행합니다")
        return None
    finally:
        sam_mask.unload()
        if moved:
            model = model.to(DEVICE)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        log(f"  [SAM] {time.time()-t0:.1f}s")


# ---------------- 엔드포인트 ----------------
@app.route("/health", methods=["GET"])
def health():
    free_gb = 0
    if DEVICE == "cuda":
        try:
            free_gb = round(torch.cuda.mem_get_info()[0] / 1e9, 1)
        except Exception:
            pass
    return jsonify({
        "alive": True,
        "busy": BUSY["now"],
        "device": DEVICE,
        "gpu": HW.get("gpu"),
        "vram_gb": round(VRAM, 1) if VRAM else HW.get("vram_gb", 0),
        "vram_free_gb": free_gb,
        "version": 2,
        "max_res": MAX_RES,
        "max_imgs": 0,                 # 0 = 제한 없음
        "auto_group": guess_group(MAX_RES),
        "recommend_offload": True,     # v2는 항상 노트북이 계산
        "sam": SAM_OK,
        "sam_reason": SAM_WHY,
        "poses": True,                 # poses.json 기반 스케일 지원
        "hq": True,                    # /reconstruct_hq (COLMAP 정밀) 지원
    })


@app.route("/capacity", methods=["POST"])
def capacity():
    """쪼개지 않고 한 번에 들어가는 최대 프레임 수를 '실측'한다.
    사진 1장을 N장으로 복제해서 실제로 추론을 돌려보고, OOM 나는 지점을 찾는다.
    (내용이 같아도 메모리 사용량은 같으므로 용량 측정에는 문제없다)"""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "이미지가 없습니다"}), 400
    res = min(int(request.form.get("res", MAX_RES)), MAX_RES)
    hard_cap = int(request.form.get("cap", 80))

    tmpdir = tempfile.mkdtemp()
    logs = []

    def log(m):
        print(m, flush=True); logs.append(str(m))

    def attempt(n, base):
        imgs = base.repeat(n, 1, 1, 1).to(DEVICE)
        try:
            with torch.no_grad():
                if DEVICE == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        pred = model(imgs)
                else:
                    pred = model(imgs)
            del pred
            return True
        finally:
            del imgs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    try:
        p = os.path.join(tmpdir, "s.jpg")
        files[0].save(p)
        base = prep_images([p], res)
        log(f"[용량 측정] 해상도 {res} · 프레임 {tuple(base.shape[-2:])} "
            f"({base.shape[-2]*base.shape[-1]/1000:.0f}k 픽셀)")

        GPU_LOCK.acquire()
        BUSY["now"] = True
        try:
            ok, bad = 0, None
            for n in (2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80):
                if n > hard_cap:
                    break
                t0 = time.time()
                try:
                    attempt(n, base)
                    ok = n
                    log(f"  {n:>3}장  OK   ({time.time()-t0:.1f}s)")
                except Exception as e:
                    if not is_oom(e):
                        raise
                    bad = n
                    log(f"  {n:>3}장  메모리 부족")
                    break
            # 성공/실패 사이를 이분 탐색으로 좁힘
            if bad is not None:
                lo, hi = ok, bad
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    try:
                        attempt(mid, base)
                        lo = mid
                        log(f"  {mid:>3}장  OK")
                    except Exception as e:
                        if not is_oom(e):
                            raise
                        hi = mid
                        log(f"  {mid:>3}장  메모리 부족")
                ok = lo
        finally:
            BUSY["now"] = False
            GPU_LOCK.release()

        log(f"[결과] 한 번에 처리 가능한 최대: {ok}장 @ {res}")
        return jsonify({"ok": True, "max_frames": ok, "res": res,
                        "estimate_was": guess_group(res), "log": logs})
    except Exception as e:
        import traceback; traceback.print_exc()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return jsonify({"error": str(e), "log": logs}), 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    t_start = time.time()
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "이미지가 없습니다"}), 400

    f = request.form
    res = min(int(f.get("res", MAX_RES)), MAX_RES)
    # 기본은 '전부 한 번에'. 체이닝은 성능 기법이 아니라 메모리 타협이라,
    # 들어가면 단일 패스가 항상 낫다. 안 들어가면 아래에서 자동으로 쪼갠다.
    group = int(f.get("group", 0)) or len(files)
    overlap = int(f.get("overlap", DEF_OVERLAP))
    # 이 둘은 '배경을 걷어내려고' 있는 필터다. SAM 마스크가 배경을 이미 지웠다면
    # 중복이고, 남은 물체까지 깎아낸다. 실측(기어 달린 모터, 32장):
    #   dist 50 / conf 25 → 51,535점, 형상이 납작해짐 (두께비 0.246)
    #   dist 100 / conf 90 → 261,092점, 입체 형상 살아남 (두께비 0.551)
    # 그래서 마스크가 실제로 잡혔을 때만 완화값으로 간다. 클라이언트가 값을
    # 명시하면 그 값을 그대로 존중한다.
    dist_given = f.get("dist_pct") is not None
    conf_given = f.get("conf_pct") is not None
    dist_pct = float(f.get("dist_pct", 50))
    conf_pct = float(f.get("conf_pct", 25))
    SAM_DIST_PCT, SAM_CONF_PCT = 100.0, 90.0
    ref_mm = float(f.get("ref_mm", 0))
    ref_mode = f.get("ref_mode", "d")
    max_dist_mm = float(f.get("max_dist_mm", 0))
    floor = float(f.get("floor", 0))
    noise_std = float(f.get("noise_std", 0))
    max_points = int(f.get("max_points", 800_000))     # 0 = 상한 없음
    sam_prompt = (f.get("sam_prompt") or "").strip()
    sam_size = int(f.get("sam_size", 0)) or None
    # sam=1 이면 프롬프트 없이도 중앙 박스 방식으로 동작한다
    use_sam = bool(sam_prompt) or f.get("sam") == "1"
    use_tsdf = f.get("tsdf") == "1"          # 깊이맵 TSDF 융합으로 메쉬까지
    tsdf_div = int(f.get("tsdf_div", 384))
    tsdf_trunc = float(f.get("tsdf_trunc", 4.0))

    poses_xyz = None
    up_axis = None
    if f.get("poses_json"):
        try:
            doc = json.loads(f["poses_json"])
            poses_xyz = np.array([p["xyz"] for p in doc["poses"]], dtype=float)
            up_axis = doc.get("up")
        except Exception as e:
            poses_xyz = None
            print(f"[!] poses_json 해석 실패: {e}", flush=True)

    tmpdir = tempfile.mkdtemp()
    logs = []

    def log(msg):
        print(msg, flush=True)
        logs.append(str(msg))

    try:
        paths = []
        for i, fl in enumerate(files):          # 장수 제한 없음
            p = os.path.join(tmpdir, f"img_{i:03d}.jpg")
            fl.save(p)
            paths.append(p)
        log(f"[요청] {len(paths)}장 @ {res} (묶음 {group})")

        # 비율이 섞이면 load_and_preprocess_images 가 '전 프레임'을 흰색으로 패딩해
        # 캔버스를 키운다. 그 여백에서 나온 쓰레기 점이 결과를 통째로 망치는데,
        # vggt 쪽 경고는 서버 콘솔로만 나가서 클라이언트는 영영 못 본다.
        # (실측 사례: 폴더에 남은 preview.png 한 장이 섞여 점 2.4배 · 시간 24배)
        try:
            from PIL import Image
            ars = []
            for p in paths:
                with Image.open(p) as im:
                    ars.append(im.size[0] / im.size[1])
            if ars and max(ars) / min(ars) > 1.05:
                names = [f.filename for f in files]
                common = max(set(round(a, 2) for a in ars),
                             key=lambda v: sum(1 for a in ars if round(a, 2) == v))
                odd = [names[i] for i, a in enumerate(ars) if round(a, 2) != common]
                log(f"  [!!] 가로세로 비율이 다른 파일 {len(odd)}개가 섞였습니다: "
                    f"{', '.join(odd[:5])}")
                log("       모든 프레임이 흰색으로 패딩됩니다 — 결과가 크게 나빠집니다. "
                    "사진이 아닌 파일을 폴더에서 빼세요.")
        except Exception:
            pass

        if BUSY["now"]:
            log("  [대기] 이전 계산이 끝나기를 기다립니다...")
        GPU_LOCK.acquire()
        BUSY["now"] = True
        try:
            masks = run_sam(paths, sam_prompt, sam_size, log) if use_sam else None
            if masks is not None and any(m is not None for m in masks):
                # 마스크가 실제로 잡힌 경우에만. SAM이 실패하면 masks=None 이므로
                # 배경이 그대로 남고, 그때는 원래 필터가 여전히 필요하다.
                if not dist_given:
                    dist_pct = SAM_DIST_PCT
                if not conf_given:
                    conf_pct = SAM_CONF_PCT
                if not (dist_given and conf_given):
                    log(f"  [필터 완화] SAM이 배경을 걸렀으므로 "
                        f"거리 {dist_pct:.0f}% · 신뢰도 {conf_pct:.0f}% 로 진행합니다 "
                        f"(고정하려면 dist_pct·conf_pct를 직접 주세요)")
            if masks is not None and floor > 0 and f.get("floor_keep") != "1":
                log("  [바닥평면 제거 끔] SAM 마스크가 이미 배경을 걸렀습니다 "
                    "(강제하려면 floor_keep=1)")
                floor = 0.0
            mesh_b64 = None
            if use_tsdf:
                if group < len(paths):
                    log("  [TSDF 건너뜀] 체이닝 중에는 아직 지원하지 않습니다 "
                        "(group을 사진 장수 이상으로 두세요)")
                else:
                    try:
                        _m = reconstruct_tsdf(paths, res, masks, conf_pct, log,
                                              voxel_div=tsdf_div,
                                              trunc_mult=tsdf_trunc)
                        import open3d as _o3d
                        _tmp = os.path.join(tmpdir, "mesh.ply")
                        _o3d.io.write_triangle_mesh(_tmp, _m)
                        with open(_tmp, "rb") as _fh:
                            mesh_b64 = base64.b64encode(_fh.read()).decode()
                    except Exception as _e:
                        log(f"  [TSDF 실패] {_e} → 점군만 반환합니다")
            (P, C, D, cams), used_res, used_group = reconstruct_adaptive(
                paths, res, group, overlap, dist_pct, conf_pct, masks, log)
            del masks
            P, C, cams, scale, dims, sinfo = postprocess(
                P, C, D, cams, ref_mm, ref_mode, max_dist_mm, floor, noise_std,
                max_points, poses_xyz, log)
        finally:
            BUSY["now"] = False
            GPU_LOCK.release()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - t_start
        log(f"[완료] 점 {len(P)}, {elapsed:.1f}s")

        buf = io.BytesIO()
        np.savez_compressed(buf, points=P.astype(np.float32), colors=C,
                            cams=cams.astype(np.float32))
        buf.seek(0)
        return jsonify({
            "ok": True, "n_points": int(len(P)),
            "elapsed_sec": round(elapsed, 2), "device": DEVICE,
            "res": used_res, "group": used_group, "n_images": len(paths),
            "scaled": scale is not None, "dims_mm": dims,
            "scale_source": sinfo["source"],
            "fit_rms_mm": sinfo["rms_mm"], "fit_max_mm": sinfo["max_mm"],
            "up": up_axis if sinfo["source"] == "poses" else None,
            "sam_used": bool(use_sam),
            "mesh_b64": mesh_b64,
            "log": logs,
            "npz_b64": base64.b64encode(buf.read()).decode(),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": str(e), "log": logs}), 500
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route("/reconstruct_hq", methods=["POST"])
def reconstruct_hq():
    """COLMAP 정밀 경로. 젯슨이 사진을 보내면 형상·STL까지 만들어 준다.

    /reconstruct 와의 차이 — 그쪽은 VGGT 9초짜리 미리보기고, 이쪽은 수십 분이다.
    실측(2026-08-16, 32장): VGGT 는 평평한 판을 5.13% 휘게 복원하고 그 왜곡이
    깊이맵 자체에 있어 후처리로 못 편다. 치수를 쓸 거면 이쪽을 써야 한다.

    사진은 서버 디스크에 저장되고, 결과 경로를 돌려준다. 점군/메쉬는 수십 MB라
    본문에 실어 보내지 않는다 (필요하면 /result 로 따로 받는다).
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "이미지가 없습니다"}), 400
    f = request.form
    sam_prompt = (f.get("sam_prompt") or "").strip()
    max_size = int(f.get("max_size", 1600))
    depth = int(f.get("depth", 9))
    smooth = int(f.get("smooth", 10))

    stamp = time.strftime("%m%d_%H%M%S")
    base = WORK / "hq" / stamp
    base.mkdir(parents=True, exist_ok=True)
    logs = []

    def log(m):
        print(m, flush=True); logs.append(str(m))

    try:
        for i, fl in enumerate(files):
            fl.save(str(base / f"img_{i:03d}.jpg"))
        log(f"[HQ] {len(files)}장 수신 → {base}")

        if BUSY["now"]:
            log("  [대기] 이전 계산이 끝나기를 기다립니다...")
        GPU_LOCK.acquire()
        BUSY["now"] = True
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import colmap_pipe
            res = colmap_pipe.run_all(str(base), sam_prompt, max_size=max_size,
                                      depth=depth, smooth=smooth, log=log)
        finally:
            BUSY["now"] = False
            GPU_LOCK.release()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return jsonify({"ok": True, "folder": str(base),
                        "object_ply": res["object_ply"], "mesh_ply": res["mesh_ply"],
                        "mesh_stl": res["mesh_stl"], "report": res["report"],
                        "log": logs})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "folder": str(base), "log": logs}), 500


@app.route("/result", methods=["GET"])
def result():
    """/reconstruct_hq 가 만든 파일 하나를 내려준다. ?path=<서버가 알려준 경로>"""
    from flask import send_file
    p = request.args.get("path", "")
    rp = os.path.realpath(p)
    if not rp.startswith(os.path.realpath(str(WORK))) or not os.path.isfile(rp):
        return jsonify({"error": "허용되지 않은 경로입니다"}), 400
    return send_file(rp, as_attachment=True)


if __name__ == "__main__":
    print("\n서버 시작: 포트 5000")
    print("젯슨에서 실행:  python3 scan.py   (IP 입력 불필요)")
    print("종료하려면 Ctrl+C\n")
    # 이 PC는 공인 IP가 NIC에 직접 붙어 있다. 0.0.0.0 은 인터넷에 그대로 열린다.
    # 젯슨이 LAN으로 붙어야 해서 기본값은 그대로 두되, 혼자 테스트할 때는
    #   set VGGT_HOST=127.0.0.1
    # 로 로컬에만 열 수 있게 한다.
    host = os.environ.get("VGGT_HOST", "0.0.0.0")
    print(f"바인드: {host}:5000")
    app.run(host=host, port=5000, threaded=True)
