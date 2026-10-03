"""
런처 — 설치 상태를 보고 알아서 설치하거나 그냥 실행

  * 처음이면: 설치 → 검증 → 실행
  * 이미 되어 있으면: 바로 실행
  * 뭔가 깨졌으면: 뭐가 없는지 알려주고 그것만 다시 설치

'VGGT 서버.bat' 이 이 파일을 부릅니다. 직접 실행할 일은 없습니다.
  --check   상태만 보고 아무것도 하지 않음
"""
import os, sys, json, socket, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path.home() / "vggt_server"
CFG = WORK / "config.json"
VENV = WORK / "venv"
MODEL = WORK / "VGGT-1B" / "model.safetensors"
PORT = 5000
NEED = ["torch", "flask", "trimesh", "sklearn", "numpy", "PIL"]

LINE = "=" * 60


def venv_python():
    if CFG.exists():
        try:
            p = json.loads(CFG.read_text(encoding="utf-8")).get("venv_python", "")
            if p and Path(p).exists():
                return Path(p)
        except Exception:
            pass
    p = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return p if p.exists() else None


def probe_venv(py):
    """가상환경 안에서 필요한 패키지가 다 되는지 확인 → (ok, 정보dict)"""
    # % 포맷을 쓰면 생성할 코드 안의 '%d.%d' 까지 바깥 포맷이 먹어버린다.
    # 문자열 이어붙이기로만 만든다.
    code = (
        "import json,sys\n"
        "r={'py':str(sys.version_info[0])+'.'+str(sys.version_info[1]),'missing':[]}\n"
        "for m in " + repr(NEED) + ":\n"
        "    try: __import__(m)\n"
        "    except Exception: r['missing'].append(m)\n"
        "try:\n"
        "    import torch; r['torch']=torch.__version__; r['cuda']=torch.cuda.is_available()\n"
        "    r['gpu']=torch.cuda.get_device_name(0) if r['cuda'] else None\n"
        "except Exception: r['torch']=None; r['cuda']=False; r['gpu']=None\n"
        "print(json.dumps(r))\n"
    )
    try:
        r = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=180)
        return (r.returncode == 0 and bool(r.stdout.strip())), json.loads(r.stdout.strip() or "{}")
    except Exception as e:
        return False, {"error": str(e)}


def state():
    """설치 상태 점검"""
    s = {"cfg": CFG.exists(), "venv": False, "model": MODEL.exists(),
         "missing": NEED[:], "torch": None, "cuda": False, "gpu": None, "py": None}
    py = venv_python()
    if py:
        ok, info = probe_venv(py)
        s["venv"] = ok
        s.update({k: info.get(k) for k in ("torch", "cuda", "gpu", "py") if k in info})
        s["missing"] = info.get("missing", NEED[:]) if ok else NEED[:]
    return s


def ready(s):
    return s["cfg"] and s["venv"] and s["model"] and not s["missing"]


def report(s):
    mark = lambda b: "OK  " if b else "없음"
    print(f"  설정 파일   {mark(s['cfg'])}")
    print(f"  가상환경    {mark(s['venv'])}" + (f"  (파이썬 {s['py']})" if s.get("py") else ""))
    if s.get("torch"):
        print(f"  파이토치    OK    {s['torch']}  GPU={'예 · ' + str(s['gpu']) if s['cuda'] else '아니오(CPU)'}")
    else:
        print(f"  파이토치    없음")
    print(f"  모델 가중치 {mark(s['model'])}" + ("" if s["model"] else "  (4.7GB, 처음 한 번만)"))
    if s["missing"]:
        print(f"  빠진 패키지 {', '.join(s['missing'])}")


def port_busy(port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def local_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ips.append(s.getsockname()[0]); s.close()
    except Exception:
        pass
    return ips


def run_setup():
    print("\n" + LINE)
    print("설치를 시작합니다. 처음이면 20~40분 걸립니다 (파이토치 2GB + 모델 4.7GB).")
    print("중간에 창을 닫지 마세요.")
    print(LINE + "\n")
    r = subprocess.run([sys.executable, str(HERE / "setup_laptop.py")])
    return r.returncode == 0


def make_shortcut():
    """바탕화면 바로가기. OneDrive로 옮겨진 바탕화면도 제대로 찾도록 API로 경로를 얻음"""
    bat = HERE / "VGGT-Server.bat"
    if not bat.exists():
        return None
    q = lambda p: str(p).replace("'", "''")      # PowerShell 작은따옴표 이스케이프
    ps = (
        "$d=[Environment]::GetFolderPath('Desktop');"
        "$p=Join-Path $d 'VGGT 3D 스캔 서버.lnk';"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($p);"
        f"$s.TargetPath='{q(bat)}';"
        f"$s.WorkingDirectory='{q(HERE)}';"
        "$s.Description='VGGT 3D 스캔 계산 서버';$s.Save();"
        "Write-Output $p"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
        out = (r.stdout or "").strip()
        return out if r.returncode == 0 and out else None
    except Exception:
        return None


def run_server(py):
    print("\n" + LINE)
    print("서버를 시작합니다.  이 창을 닫으면 서버가 꺼집니다.")
    for ip in local_ips():
        print(f"  이 노트북 주소: {ip}:{PORT}   (젯슨은 알아서 찾습니다)")
    print("  끄려면 Ctrl+C")
    print(LINE + "\n")
    try:
        return subprocess.run([str(py), str(HERE / "server.py")]).returncode
    except KeyboardInterrupt:
        print("\n서버를 껐습니다.")
        return 0


def run_localtest(args):
    """가상환경 파이썬으로 localtest.py 실행 (VGGT-Test.bat 이 부름)"""
    py = venv_python()
    if py is None:
        print("[!] 가상환경이 없습니다. 먼저 VGGT-Server.bat 을 실행해 설치하세요.")
        return 1
    script = HERE / "localtest.py"
    if not script.exists():
        print(f"[!] {script} 가 없습니다.")
        return 1
    return subprocess.run([str(py), str(script), *args]).returncode


def sam_setup():
    """SAM 3 로그인 + 가중치 미리 받기 (설치를 이미 끝낸 뒤 승인이 났을 때)"""
    py = venv_python()
    if py is None:
        print("[!] 가상환경이 없습니다. 먼저 VGGT-Server.bat 으로 설치하세요.")
        return 1

    print("\n" + LINE)
    print("  SAM 3 설정")
    print(LINE)
    print("\n먼저 아래 두 가지를 브라우저에서 끝내주세요:")
    print("  1) https://huggingface.co/facebook/sam3  →  접근 신청 (약관 동의)")
    print("     '가입'만으로는 안 되고, 이 페이지에서 신청을 눌러야 합니다.")
    print("  2) https://huggingface.co/settings/tokens →  New token → Read 권한으로 생성")
    print("\n생성한 토큰(hf_ 로 시작)을 여기에 붙여넣으세요.")

    try:
        import getpass
        token = getpass.getpass("  토큰 (입력해도 화면에 안 보입니다): ").strip()
    except Exception:
        token = input("  토큰: ").strip()
    if not token:
        print("  취소했습니다.")
        return 1

    env = dict(os.environ)
    env["HF_TOKEN_INPUT"] = token
    env["HF_HUB_OFFLINE"] = "0"

    print("\n[1/2] 로그인")
    code = ("import os\n"
            "from huggingface_hub import login\n"
            "login(token=os.environ['HF_TOKEN_INPUT'], add_to_git_credential=False)\n"
            "print('LOGIN_OK')\n")
    r = subprocess.run([str(py), "-c", code], env=env, capture_output=True, text=True)
    if "LOGIN_OK" not in (r.stdout or ""):
        print(f"  [!] 로그인 실패: {(r.stderr or r.stdout or '').strip()[:300]}")
        return 1
    print("  로그인 완료 (이 계정으로 이 PC 전체에서 쓰입니다)")

    print("\n[2/2] SAM 3 가중치 받기 — 몇 GB라 시간이 걸립니다")
    code2 = ("import os\n"
             "os.environ['HF_HUB_OFFLINE']='0'\n"
             "from transformers import Sam3Model, Sam3Processor\n"
             "Sam3Model.from_pretrained('facebook/sam3')\n"
             "Sam3Processor.from_pretrained('facebook/sam3')\n"
             "print('SAM3_OK')\n")
    r2 = subprocess.run([str(py), "-c", code2], env=env, capture_output=True, text=True)
    if "SAM3_OK" in (r2.stdout or ""):
        print("\n  완료! 이제 스캔할 때 SAM 물체 설명을 넣으면 물체만 남습니다.")
        return 0

    err = (r2.stderr or "").strip()
    print(f"\n  [!] 가중치를 못 받았습니다.")
    if "401" in err or "403" in err or "gated" in err.lower() or "awaiting" in err.lower():
        print("      접근 승인이 아직 안 났습니다. facebook/sam3 페이지에서")
        print("      신청 상태를 확인하고, 승인된 뒤 이 파일을 다시 실행하세요.")
    else:
        print(f"      {err[:400]}")
    return 1


def main():
    if "--sam-setup" in sys.argv:
        return sam_setup()
    if "--test" in sys.argv:
        i = sys.argv.index("--test")
        return run_localtest(sys.argv[i + 1:])

    check_only = "--check" in sys.argv
    print(LINE)
    print("  VGGT 3D 스캔 — 노트북 계산 서버")
    print(LINE)

    missing_files = [f for f in ("setup_laptop.py", "server.py") if not (HERE / f).exists()]
    if missing_files:
        print(f"\n[!] 같은 폴더에 {', '.join(missing_files)} 가 없습니다.")
        print(f"    받은 파일을 전부 한 폴더에 두고 실행하세요.")
        print(f"    현재 폴더: {HERE}")
        return 1

    print("\n[상태 확인]")
    s = state()
    report(s)

    if check_only:
        print(f"\n결과: {'실행 준비 완료' if ready(s) else '설치 필요'}")
        return 0

    if not ready(s):
        print("\n설치가 필요합니다.")
        if not run_setup():
            print("\n[!] 설치가 끝나지 않았습니다. 위 메시지를 확인하세요.")
            return 1
        print("\n[재확인]")
        s = state()
        report(s)
        if not ready(s):
            print("\n[!] 아직 준비가 안 됐습니다. 위에서 '없음'인 항목이 원인입니다.")
            print("    인터넷/디스크 공간을 확인하고 다시 실행해 보세요.")
            return 1
        try:
            if input("\n바탕화면에 바로가기를 만들까요? (Y/n): ").strip().lower() not in ("n", "no"):
                lnk = make_shortcut()
                print(f"  만들었습니다: {lnk}" if lnk else "  (바로가기 생성 실패 — 무시해도 됩니다)")
        except EOFError:
            pass

    if not s["cuda"]:
        print("\n[!] GPU를 못 씁니다 → CPU로 돕니다 (매우 느림).")
        print("    NVIDIA 드라이버를 확인하거나 setup_laptop.py 를 다시 돌리세요.")

    if port_busy():
        print(f"\n[!] 포트 {PORT}이 이미 쓰이고 있습니다.")
        print("    서버가 이미 켜져 있을 수 있습니다. 기존 창을 확인하세요.")
        try:
            if input("    그래도 계속할까요? (y/N): ").strip().lower() != "y":
                return 0
        except EOFError:
            return 0

    return run_server(venv_python())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        sys.exit(0)
