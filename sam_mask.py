"""
SAM 3 물체 마스크 (노트북에서 실행)

중요 — 이 마스크는 VGGT '앞'이 아니라 '뒤'에 씁니다.
  배경을 지운 이미지를 VGGT에 넣으면 프레임 간 정합 단서(배경 텍스처)가 사라져서
  카메라 포즈 추정이 무너집니다. 물체가 무광/무늬 없으면 특히 심합니다.
  그래서 여기서는 마스크만 만들고, 점군 단계에서 픽셀 대응으로 걸러냅니다.
  VGGT 출력 world_points가 (H,W,3) 픽셀당 3D 점이라 2D 마스크가 1:1로 맞습니다.

준비물 (노트북에서 1회):
  1) https://huggingface.co/facebook/sam3 에서 체크포인트 접근 신청 (승인제)
  2) pip install "transformers>=5" accelerate
  3) hf auth login
"""
import numpy as np

MODEL_ID = "facebook/sam3"
_M = {"model": None, "proc": None}


def available():
    """(사용가능?, 사유) — 서버가 시작할 때 한 번 물어보는 용도"""
    try:
        from transformers import Sam3Model, Sam3Processor  # noqa: F401
    except Exception as e:
        return False, f"transformers에 Sam3가 없습니다 ({type(e).__name__}). pip install -U transformers"
    return True, "ok"


def load(device="cuda", image_size=None, dtype=None):
    """모델 적재. image_size를 낮추면 메모리는 줄지만 정확도가 떨어집니다(기본 1008)."""
    if _M["model"] is not None:
        return _M["model"], _M["proc"]
    import os, torch
    from transformers import Sam3Model, Sam3Processor

    # server.py가 VGGT를 오프라인으로 묶어두는데(HF_HUB_OFFLINE=1), SAM 3는
    # 처음 한 번 받아와야 하므로 이 구간에서만 잠시 푼다.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        return _build(device, image_size, dtype, torch, Sam3Model, Sam3Processor)
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline


def _build(device, image_size, dtype, torch, Sam3Model, Sam3Processor):
    kw = {}
    if dtype is None and device == "cuda":
        dtype = torch.float16
    if dtype is not None:
        kw["dtype"] = dtype
    if image_size:
        from transformers import Sam3Config
        cfg = Sam3Config.from_pretrained(MODEL_ID)
        cfg.image_size = int(image_size)
        kw["config"] = cfg
        proc = Sam3Processor.from_pretrained(
            MODEL_ID, size={"height": int(image_size), "width": int(image_size)})
    else:
        proc = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID, **kw).to(device).eval()
    _M["model"], _M["proc"] = model, proc
    return model, proc


def unload():
    """VGGT에게 VRAM을 돌려주기 위해 반드시 호출"""
    import torch, gc
    _M["model"] = None
    _M["proc"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def center_box(w, h, frac=0.6):
    """화면 중앙 frac 비율 영역 (xyxy, 픽셀)"""
    cw, ch = w * frac / 2.0, h * frac / 2.0
    return [w / 2 - cw, h / 2 - ch, w / 2 + cw, h / 2 + ch]


def _bbox_of(mask, pad=0.15):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return [max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)]


def segment(paths, prompt="", device="cuda", image_size=None,
            score_thresh=0.5, merge=False, box_frac=0.6, log=print):
    """사진 경로들 → 프레임별 bool 마스크 리스트 (원본 사진 크기).

    prompt 가 비어 있으면 '중앙 박스' 방식으로 동작한다. 물체를 화면 중앙에
    두고 찍는 리그에서는 물체 이름을 몰라도 되고, 주변에 같은 종류의 물체가
    있어도 엉뚱한 걸 잡지 않는다. 첫 프레임에서 마스크를 얻으면 그 bbox를
    다음 프레임의 박스로 물려줘서, 물체가 조금씩 움직여도 따라간다.

    프롬프트를 '|' 로 여러 개 줄 수 있고, 그때는 각 프롬프트의 마스크를 합집합으로
    묶는다. 부품이 여럿인 물체는 한 낱말로 전부 안 잡히기 때문이다.
    실측 예: 모터 위에 기어가 달린 물체에서
      'blue gear'            → 기어만 4.6%
      'the silver metal object' → 모터 몸체만 1.4%
      'blue gear|the silver metal object' → 둘 다
    (프롬프트 수만큼 추론이 늘어나므로 필요한 만큼만 쓸 것)

    마스크를 못 찾은 프레임은 None (호출하는 쪽에서 '전체 통과'로 처리)."""
    import torch
    from PIL import Image

    model, proc = load(device, image_size)
    masks = []
    n_hit = 0
    box = None
    prompts = [s.strip() for s in prompt.split("|") if s.strip()] if prompt else []
    if prompts:
        mode = "텍스트 " + " + ".join(f"'{s}'" for s in prompts)
    else:
        mode = f"중앙 박스 {int(box_frac*100)}%"
    log(f"  [SAM 3] {mode}")

    def to_model(bf):
        """프로세서는 항상 fp32 텐서를 준다. 그런데 8GB에서는 모델을 fp16으로 올리므로
        그대로 넣으면 첫 행렬곱에서 죽는다:
          'mat1 and mat2 must have the same dtype, but got Float and Half'
        BatchFeature.to(dtype)은 실수 텐서만 바꾸고 정수 텐서(original_sizes 등)는
        건드리지 않으므로 이 한 줄로 충분하다."""
        bf = bf.to(model.device)
        if model.dtype != torch.float32:
            bf = bf.to(model.dtype)
        return bf

    def one(img, text=None, use_box=None):
        """프롬프트 하나(또는 박스 하나) → (마스크, 점수, 인스턴스 수). 없으면 (None,0,0)"""
        if text is not None:
            inputs = to_model(proc(images=img, text=text, return_tensors="pt"))
        else:
            inputs = to_model(proc(images=img, input_boxes=[[use_box]],
                                   input_boxes_labels=[[1]], return_tensors="pt"))
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_instance_segmentation(
            out, threshold=score_thresh, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        mk, sc = res["masks"], res["scores"]
        if len(mk) == 0:
            return None, 0.0, 0
        arr = mk.detach().cpu().numpy() if hasattr(mk, "detach") else np.asarray(mk)
        if arr.dtype != bool:
            arr = arr > 0.5
        scores = sc.detach().cpu().numpy() if hasattr(sc, "detach") else np.asarray(sc)
        if merge:
            return arr.any(axis=0), float(scores.max()), len(arr)
        k = int(np.argmax(scores))
        return arr[k], float(scores[k]), len(arr)

    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        m = None; best = 0.0; n_inst = 0
        if prompts:
            for text in prompts:              # 여러 프롬프트는 합집합
                mi, si, ni = one(img, text=text)
                if mi is None:
                    continue
                m = mi if m is None else (m | mi)
                best = max(best, si); n_inst += ni
        else:
            if box is None:
                box = center_box(img.width, img.height, box_frac)
            m, best, n_inst = one(img, use_box=box)

        if m is None:
            masks.append(None)
            log(f"    [{i+1}/{len(paths)}] 마스크 없음 — 이 프레임은 필터 안 함")
            continue

        masks.append(m)
        n_hit += 1
        if not prompts:                      # 다음 프레임 박스를 이 마스크로 갱신
            nb = _bbox_of(m)
            if nb is not None:
                box = nb
        if i == 0 or (i + 1) % 10 == 0 or i == len(paths) - 1:
            log(f"    [{i+1}/{len(paths)}] 인스턴스 {n_inst}개, 점수 {best:.2f}, "
                f"물체 비율 {m.mean()*100:.1f}%")

    log(f"  [SAM 3] {n_hit}/{len(paths)} 프레임에서 마스크 확보")
    return masks


def resize_mask(mask, H, W):
    """원본 크기 bool 마스크 → VGGT 입력 텐서 크기 (nearest)"""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(mask.astype(np.uint8))[None, None].float()
    t = F.interpolate(t, size=(H, W), mode="nearest")
    return t[0, 0].numpy() > 0.5
