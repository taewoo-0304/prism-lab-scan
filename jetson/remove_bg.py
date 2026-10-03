"""
사진에서 물체만 남기고 배경 제거 (rembg)
사용법:
  python3 remove_bg.py <입력폴더> [출력폴더] [배경색]
   예) python3 remove_bg.py ~/vggt/scan_0723_211418
   → ~/vggt/scan_0723_211418_nobg/ 에 물체만 남은 사진 저장
배경색: white(기본) / black / green
  * 이후 이 _nobg 폴더를 chain2.py나 offload.py에 넣으면 됩니다.
"""
import os, sys, glob, time
import numpy as np
from PIL import Image

BG = {"white": (255, 255, 255), "black": (0, 0, 0), "green": (0, 177, 64)}


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 remove_bg.py <입력폴더> [출력폴더] [white/black/green]")
        return
    src = os.path.expanduser(sys.argv[1])
    dst = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else src.rstrip("/") + "_nobg"
    bg_name = sys.argv[3] if len(sys.argv) > 3 else "white"
    bg = BG.get(bg_name, (255, 255, 255))
    os.makedirs(dst, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(src, "*.jpg")) +
                   glob.glob(os.path.join(src, "*.png")))
    if not paths:
        print(f"[!] {src} 에 사진이 없습니다"); return
    print(f"사진 {len(paths)}장 → 배경 제거 → {dst}  (배경: {bg_name})")

    from rembg import remove, new_session
    session = new_session("u2net")   # 모델 1회 로딩

    t0 = time.time()
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        cut = remove(img, session=session)          # RGBA
        alpha = cut.split()[3]
        # 물체 비율 체크 (너무 작으면 마스킹 실패 가능성)
        a = np.array(alpha)
        frac = (a > 128).mean() * 100
        canvas = Image.new("RGB", cut.size, bg)
        canvas.paste(cut, mask=alpha)
        out = os.path.join(dst, f"img_{i:03d}.jpg")
        canvas.save(out, quality=95)
        flag = "  ← 물체가 너무 작음(확인)" if frac < 3 else ""
        print(f"  [{i+1}/{len(paths)}] 물체 {frac:.0f}%{flag}")
    print(f"완료 ({time.time()-t0:.0f}s) → {dst}")
    print(f"\n이제 이 폴더로 복원하세요:")
    print(f"  python3 chain2.py")
    print(f"  >>> {dst} 4 224 2 50 0 25 1.5     (바닥제거 0 — 배경 없으니 불필요)")


if __name__ == "__main__":
    main()
