"""바탕화면에서 가장 최근 촬영 폴더를 찾아 경로 한 줄을 찍는다.

배치가 폴더를 안 받았을 때 쓴다. 촬영 폴더는 jpg 가 여러 장 든 폴더다.
결과 폴더(_scan, _out 등)나 빈 폴더는 제외한다.
"""
import os, sys, glob

MIN_JPG = 5


def newest(root=None):
    root = root or os.path.join(os.path.expanduser("~"), "Desktop")
    best, best_t = None, -1.0
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or name.startswith((".", "_")):
            continue
        jpgs = glob.glob(os.path.join(d, "*.jpg"))
        if len(jpgs) < MIN_JPG:
            continue
        t = max(os.path.getmtime(p) for p in jpgs)
        if t > best_t:
            best, best_t = d, t
    return best


if __name__ == "__main__":
    d = newest(sys.argv[1] if len(sys.argv) > 1 else None)
    if d:
        print(d)
        sys.exit(0)
    sys.exit(1)
