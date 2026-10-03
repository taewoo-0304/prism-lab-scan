"""VGGT 서버가 뜰 때까지 기다린다. 뜨면 0, 시간 초과면 1.

    python _wait_server.py [초]
"""
import sys, time, requests

URL = "http://127.0.0.1:5000/health"


def wait(timeout=200.0, step=4.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(URL, timeout=3).json().get("alive"):
                return True
        except Exception:
            pass
        time.sleep(step)
    return False


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
    sys.exit(0 if wait(t) else 1)
