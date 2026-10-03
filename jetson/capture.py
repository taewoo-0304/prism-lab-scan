"""
촬영 스크립트 (단독 사용) — 가상환경 없이 시스템 파이썬으로 실행
  python3 capture.py
  SPACE = 촬영,  q = 종료
* scan.py를 쓰면 이건 따로 안 돌려도 됩니다 (scan.py 안에 촬영이 포함됨).
"""
import cv2, os, sys, time

folder = sys.argv[1] if len(sys.argv) > 1 else None
if not folder:
    name = input("저장할 폴더 이름 (예: test4): ").strip() or ("scan_" + time.strftime("%m%d_%H%M"))
    folder = os.path.expanduser(f"~/vggt/{name}")
os.makedirs(folder, exist_ok=True)
print(f"저장 위치: {folder}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] 카메라를 열 수 없습니다."); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

n = 0
print("SPACE=촬영, q=종료  |  물체 고정, 카메라 들고 한 바퀴, 시작 거리 일정하게")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    disp = frame.copy()
    cv2.putText(disp, f"{n} shots  [SPACE]capture  [q]done", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("capture", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        p = os.path.join(folder, f"img_{n:03d}.jpg")
        cv2.imwrite(p, frame)
        n += 1
        print(f"  촬영 {n}장")
    elif k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print(f"완료: {n}장 → {folder}")
