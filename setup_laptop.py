"""
노트북 완전 자동 세팅 — 파이썬 버전 고정 + 전부 자동 다운로드
  * 파이썬 3.11/3.12 만 사용 (PyTorch 공식 지원 범위). 3.13이면 3.12를 자동 설치
  * git 없으면 자동 설치 (실패해도 zip으로 대체)
  * PyTorch는 여러 CUDA 채널 자동 시도 → 성공할 때까지
  * 설치 후 반드시 검증. 실패하면 조용히 넘어가지 않고 멈춤
사용법:  python setup_laptop.py
"""
import os, sys, subprocess, shutil, platform, json, venv, tempfile, urllib.request, re
from pathlib import Path

WORK = Path.home() / "vggt_server"
VENV = WORK / "venv"
VGGT_DIR = WORK / "vggt"
MODEL_DIR = WORK / "VGGT-1B"
IS_WIN = platform.system() == "Windows"

# ===== 고정 사양 (젯슨에서 배운 교훈: 버전 안 맞으면 다 깨진다) =====
PY_OK = [(3, 12), (3, 11), (3, 10)]          # PyTorch가 확실히 지원하는 버전 (우선순위)
PY_INSTALL_VER = "3.12.8"                     # 없을 때 자동 설치할 버전
CUDA_CHANNELS = ["cu124", "cu126", "cu121", "cu118"]   # 위에서부터 시도
GIT_WIN_URL = ("https://github.com/git-for-windows/git/releases/download/"
               "v2.47.1.windows.1/Git-2.47.1-64-bit.exe")
PY_WIN_URL = f"https://www.python.org/ftp/python/{PY_INSTALL_VER}/python-{PY_INSTALL_VER}-amd64.exe"


def run(cmd, check=True, show=True, timeout=None):
    if show:
        c = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
        print(f"  $ {c[:150]}")
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("    [!] 시간 초과")
        return False, ""
    out = (r.stdout or "").strip()
    if out and show:
        print("   ", out[:200].replace("\n", "\n    "))
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        if err and show and check:
            print("    [!]", err[:200])
        return False, out
    return True, out


def dl(url, dest, label=""):
    print(f"  다운로드: {label or url.split('/')[-1]}")
    def hook(b, bs, total):
        if total > 0:
            print(f"\r    {min(100, b*bs*100//total)}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print("\r    100%   ")


# ---------- 1. 파이썬 버전 고정 ----------
def py_version(exe):
    ok, out = run(f'"{exe}" -c "import sys;print(f\'{{sys.version_info.major}}.{{sys.version_info.minor}}\')"',
                  check=False, show=False)
    if ok and out.strip():
        try:
            a, b = out.strip().split(".")
            return (int(a), int(b))
        except Exception:
            pass
    return None


def find_good_python():
    """PyTorch가 지원하는 파이썬(3.10~3.12) 찾기"""
    cands = []
    # 현재 실행중인 파이썬
    cands.append(sys.executable)
    if IS_WIN:
        # py 런처로 설치된 버전들 확인
        ok, out = run("py -0p", check=False, show=False)
        if ok:
            for line in out.splitlines():
                m = re.search(r"(\S+python\.exe)", line, re.I)
                if m:
                    cands.append(m.group(1))
        for v in ["3.12", "3.11", "3.10"]:
            for base in [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python",
                         Path("C:/")]:
                p = base / f"Python{v.replace('.','')}" / "python.exe"
                if p.exists():
                    cands.append(str(p))
    else:
        for v in ["3.12", "3.11", "3.10"]:
            p = shutil.which(f"python{v}")
            if p:
                cands.append(p)

    seen = set()
    found = {}
    for c in cands:
        if not c or c in seen or not Path(c).exists():
            continue
        seen.add(c)
        v = py_version(c)
        if v in PY_OK and v not in found:
            found[v] = c
    for v in PY_OK:            # 우선순위대로
        if v in found:
            return found[v], v
    return None, None


def install_python():
    """파이썬 3.12 자동 설치 (윈도우)"""
    print(f"  파이썬 {PY_INSTALL_VER} 자동 설치")
    if IS_WIN:
        if shutil.which("winget"):
            print("  winget으로 시도...")
            run("winget install --id Python.Python.3.12 -e --source winget "
                "--accept-package-agreements --accept-source-agreements",
                check=False, timeout=900)
            exe, v = find_good_python()
            if exe:
                return exe, v
        try:
            tmp = Path(tempfile.gettempdir()) / "python_installer.exe"
            dl(PY_WIN_URL, tmp, f"Python {PY_INSTALL_VER}")
            print("  무인 설치 중... (2~3분)")
            run(f'"{tmp}" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1',
                check=False, timeout=1200)
            exe, v = find_good_python()
            if exe:
                return exe, v
        except Exception as e:
            print(f"    [!] {e}")
    else:
        if shutil.which("apt-get"):
            run("sudo apt-get install -y python3.12 python3.12-venv", check=False, timeout=900)
            exe, v = find_good_python()
            if exe:
                return exe, v
    return None, None


# ---------- 2. git ----------
def find_git():
    g = shutil.which("git")
    if g:
        return g
    if IS_WIN:
        for p in [r"C:\Program Files\Git\cmd\git.exe",
                  r"C:\Program Files (x86)\Git\cmd\git.exe",
                  str(Path.home() / r"AppData\Local\Programs\Git\cmd\git.exe")]:
            if Path(p).exists():
                return p
    return None


def ensure_git():
    g = find_git()
    if g:
        print(f"  git 있음")
        return g
    print("  git 없음 → 자동 설치")
    if IS_WIN:
        if shutil.which("winget"):
            run("winget install --id Git.Git -e --source winget "
                "--accept-package-agreements --accept-source-agreements",
                check=False, timeout=600)
            g = find_git()
            if g:
                return g
        try:
            tmp = Path(tempfile.gettempdir()) / "git_installer.exe"
            dl(GIT_WIN_URL, tmp, "Git")
            print("  무인 설치 중...")
            run(f'"{tmp}" /VERYSILENT /NORESTART /NOCANCEL /SP-', check=False, timeout=900)
            g = find_git()
            if g:
                return g
        except Exception as e:
            print(f"    [!] {e}")
    else:
        if shutil.which("apt-get"):
            run("sudo apt-get install -y git", check=False, timeout=600)
            g = find_git()
            if g:
                return g
    print("  git 실패 → zip 다운로드로 대체 (문제 없음)")
    return None


def get_vggt_code(git_exe):
    if VGGT_DIR.exists():
        print("  VGGT 코드 있음")
        return True
    if git_exe:
        ok, _ = run(f'"{git_exe}" clone https://github.com/facebookresearch/vggt.git "{VGGT_DIR}"',
                    check=False, timeout=900)
        if ok and VGGT_DIR.exists():
            return True
    print("  zip으로 다운로드")
    try:
        import zipfile
        tmp = Path(tempfile.gettempdir()) / "vggt.zip"
        dl("https://github.com/facebookresearch/vggt/archive/refs/heads/main.zip", tmp, "VGGT 코드")
        with zipfile.ZipFile(tmp) as z:
            root = z.namelist()[0].split("/")[0]
            z.extractall(VGGT_DIR.parent)
        (VGGT_DIR.parent / root).rename(VGGT_DIR)
        return True
    except Exception as e:
        print(f"  [!] 실패: {e}")
        return False


# ---------- 3. 하드웨어 ----------
def detect_hardware():
    info = {"os": platform.system(), "gpu": None, "vram_gb": 0, "cuda": None}
    ok, out = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                  check=False, show=False)
    if ok and out.strip():
        try:
            name, mem = [x.strip() for x in out.strip().split("\n")[0].split(",")]
            info["gpu"] = name
            info["vram_gb"] = round(float(mem.replace("MiB", "").strip()) / 1024, 1)
        except Exception:
            pass
    ok, out = run("nvidia-smi", check=False, show=False)
    if ok and "CUDA Version" in out:
        try:
            info["cuda"] = out.split("CUDA Version:")[1].split()[0]
        except Exception:
            pass
    return info


def pick_profile(vram):
    """server.py v2는 장수 제한이 없고(넘치면 자동 체이닝) 해상도는 518이 상한.
    여기 값은 '한 번에 GPU에 올릴 프레임 수' 참고치일 뿐 하드 캡이 아니다."""
    if vram >= 20: return {"max_imgs": 0, "res": 518, "group": 20, "chain": True, "tier": "최고"}
    if vram >= 12: return {"max_imgs": 0, "res": 518, "group": 12, "chain": True, "tier": "높음"}
    if vram >= 8:  return {"max_imgs": 0, "res": 518, "group": 8,  "chain": True, "tier": "중상"}
    if vram >= 6:  return {"max_imgs": 0, "res": 518, "group": 5,  "chain": True, "tier": "중간"}
    if vram >= 4:  return {"max_imgs": 0, "res": 448, "group": 4,  "chain": True, "tier": "낮음"}
    return {"max_imgs": 0, "res": 336, "group": 3, "chain": True, "tier": "CPU(느림)"}


# ---------- 4. 가상환경 (지정 파이썬으로) ----------
def venv_python():
    return str(VENV / ("Scripts/python.exe" if IS_WIN else "bin/python"))


def make_venv(base_python, force=False):
    WORK.mkdir(exist_ok=True)
    if Path(venv_python()).exists():
        v = py_version(venv_python())
        if v in PY_OK and not force:
            print(f"  가상환경 있음 (파이썬 {v[0]}.{v[1]})")
            return True
        print(f"  기존 가상환경 파이썬 {v} → 부적합. 새로 만듭니다")
        shutil.rmtree(VENV, ignore_errors=True)
    _v = py_version(base_python)
    print(f"  가상환경 생성 (파이썬 {_v[0]}.{_v[1]})" if _v else "  가상환경 생성")
    ok, _ = run(f'"{base_python}" -m venv "{VENV}"', timeout=300)
    if not ok or not Path(venv_python()).exists():
        return False
    run(f'"{venv_python()}" -m pip install --upgrade pip', show=False)
    return True


def vpip(pkgs, extra="", show=True):
    return run(f'"{venv_python()}" -m pip install {pkgs} {extra}', check=False,
               show=show, timeout=3600)[0]


def vpip_uninstall(pkgs, show=False):
    # pip install --uninstall 은 존재하지 않는 문법이라 조용히 실패했었음
    return run(f'"{venv_python()}" -m pip uninstall -y {pkgs}', check=False,
               show=show, timeout=900)[0]


# ---------- 5. 파이토치 (채널 자동 시도 + 검증) ----------
def verify_torch():
    ok, out = run(f'"{venv_python()}" -c "import torch;print(torch.__version__,torch.cuda.is_available())"',
                  check=False, show=False)
    if ok and out.strip():
        parts = out.split()
        return parts[0], parts[1] == "True"
    return None, False


def install_torch(info):
    ver, cuda = verify_torch()
    if ver and (cuda or not info["gpu"]):
        print(f"  torch 이미 정상 ({ver}, CUDA={cuda})")
        return True
    if ver and info["gpu"] and not cuda:
        print(f"  torch {ver}가 CPU판인데 GPU 있음 → 재설치")
        vpip_uninstall("torch torchvision")

    if info["gpu"]:
        # nvidia-smi CUDA 버전에 맞는 채널을 우선 시도
        chans = list(CUDA_CHANNELS)
        cu = info.get("cuda")
        if cu:
            try:
                maj, minr = cu.split(".")[:2]
                pref = f"cu{maj}{minr}"
                if pref in chans:
                    chans.remove(pref)
                chans.insert(0, pref)
            except Exception:
                pass
        for ch in chans:
            print(f"  CUDA 채널 {ch} 시도 (2~3GB)")
            vpip("torch torchvision", f"--index-url https://download.pytorch.org/whl/{ch}", show=False)
            ver, cuda = verify_torch()
            if ver and cuda:
                print(f"  성공! torch {ver} (CUDA 사용가능, 채널 {ch})")
                return True
            elif ver:
                print(f"    torch {ver} 설치됐지만 CUDA 인식 안 됨 → 다음 채널")
                vpip_uninstall("torch torchvision")
            else:
                print(f"    {ch} 실패 → 다음 채널")
        print("  [!] 모든 CUDA 채널 실패 → CPU판으로 대체")

    print("  CPU 파이토치 설치")
    vpip("torch torchvision", "--index-url https://download.pytorch.org/whl/cpu", show=False)
    ver, cuda = verify_torch()
    if ver:
        print(f"  torch {ver} (CPU)")
        return True
    print("  [!!] 파이토치 설치 완전 실패")
    return False


# ---------- 실행 ----------
if __name__ == "__main__":
    print("=" * 62)
    print("노트북 자동 세팅 (파이썬 버전 고정 + 전부 자동 다운로드)")
    print("=" * 62)

    print("\n[1/6] 파이썬 버전 확인 (PyTorch는 3.10~3.12만 지원)")
    cur = py_version(sys.executable)
    print(f"  실행중인 파이썬: {cur[0]}.{cur[1]}")
    base_py, bv = find_good_python()
    if not base_py:
        print(f"  [!] 3.10~3.12가 없음 (3.13은 PyTorch 미지원) → 자동 설치")
        base_py, bv = install_python()
        if not base_py:
            print("\n  [!!] 파이썬 자동 설치 실패.")
            print(f"     https://www.python.org/downloads/release/python-{PY_INSTALL_VER.replace('.','')}/")
            print("     에서 설치 후(Add to PATH 체크) 다시 실행하세요.")
            input("엔터로 종료...")
            sys.exit(1)
    print(f"  사용할 파이썬: {bv[0]}.{bv[1]}  ({base_py})")

    print("\n[2/6] git (없으면 자동 설치)")
    git_exe = ensure_git()

    print("\n[3/6] 하드웨어 감지")
    info = detect_hardware()
    if info["gpu"]:
        print(f"  GPU: {info['gpu']}  VRAM {info['vram_gb']}GB  CUDA {info['cuda']}")
    else:
        print("  NVIDIA GPU 없음 → CPU 모드")
    prof = pick_profile(info["vram_gb"])
    print(f"  → 프로파일: {prof['tier']} / 장수 무제한 · 해상도 {prof['res']} "
          f"(한 묶음 {prof['group']}장 기준)")

    print("\n[4/6] 가상환경")
    if not make_venv(base_py):
        print("  [!!] 가상환경 생성 실패")
        input("엔터로 종료..."); sys.exit(1)

    print("\n[5/6] 파이토치 (채널 자동 시도)")
    if not install_torch(info):
        print("\n  [!!] 파이토치 설치 실패 — 여기서 멈춥니다 (조용히 넘어가지 않음)")
        input("엔터로 종료..."); sys.exit(1)

    print("\n[6/6] VGGT + 모델")
    if not get_vggt_code(git_exe):
        input("엔터로 종료..."); sys.exit(1)
    print("  패키지 설치")
    vpip("numpy pillow huggingface_hub einops safetensors trimesh scikit-learn flask requests",
         show=False)
    print("  도면 생성용 패키지 (노트북 단독 테스트에 필요)")
    vpip("matplotlib scipy scikit-image", show=False)
    # open3d·plyfile 은 원래 목록에 없었다. 그런데 object_clean·colmap_pipe·
    # 메쉬 생성이 전부 open3d 를 쓴다. 개발 PC 에는 따로 깔려 있어서 드러나지
    # 않았을 뿐, 새 컴퓨터에서는 스캔 정리 단계에서 바로 죽는다.
    vpip("open3d plyfile", show=False)
    print("  SAM 3용 패키지 (선택 — 실패해도 나머지는 동작)")
    vpip("transformers accelerate", show=False)
    # SAM 3 가중치 미리 받기. 승인 전이거나 로그인 전이면 조용히 넘어간다
    # (서버 첫 요청 때 받아도 되지만, 그때 받으면 스캔이 한참 멈춘다)
    code = ("from transformers import Sam3Model, Sam3Processor as P;"
            "Sam3Model.from_pretrained('facebook/sam3');P.from_pretrained('facebook/sam3');"
            "print('SAM3_OK')")
    ok, out = run(f'"{venv_python()}" -c "{code}"', check=False, show=False, timeout=3600)
    if ok and "SAM3_OK" in out:
        print("  SAM 3 가중치 준비 완료")
    else:
        print("  SAM 3 가중치 없음 — 접근 승인/로그인이 필요합니다 (없어도 나머지는 동작)")
        print("     1) https://huggingface.co/facebook/sam3 에서 접근 신청")
        print(f'     2) "{VENV / ("Scripts/hf.exe" if IS_WIN else "bin/hf")}" auth login')
    if not (MODEL_DIR / "model.safetensors").exists():
        print("  모델 가중치 다운로드 (4.7GB — 오래 걸립니다)")
        code = ("from huggingface_hub import snapshot_download;"
                f"snapshot_download('facebook/VGGT-1B', local_dir=r'{MODEL_DIR}')")
        run(f'"{venv_python()}" -c "{code}"', timeout=7200)
    else:
        print("  모델 가중치 있음")

    # 최종 검증
    print("\n[검증]")
    ver, cuda = verify_torch()
    pv = py_version(venv_python())
    print(f"  파이썬 {pv[0]}.{pv[1]} / torch {ver} / CUDA {cuda}")
    model_ok = (MODEL_DIR / "model.safetensors").exists()
    print(f"  모델 가중치: {'OK' if model_ok else '없음'}")
    if info["gpu"] and not cuda:
        print("  [!] GPU가 있는데 CUDA를 못 씁니다 → NVIDIA 드라이버를 최신으로 업데이트하세요")

    cfg = {"hardware": info, "profile": prof, "vggt_dir": str(VGGT_DIR),
           "model_dir": str(MODEL_DIR), "venv_python": venv_python(),
           "python": f"{pv[0]}.{pv[1]}", "torch": ver, "cuda_ok": cuda}
    (WORK / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    here = Path(__file__).parent.resolve()
    if IS_WIN:
        bat = WORK / "run_server.bat"
        bat.write_text(f'@echo off\r\n"{venv_python()}" "{here / "server.py"}"\r\npause\r\n',
                       encoding="utf-8")
        launch = f"{bat}   ← 더블클릭"
    else:
        sh = WORK / "run_server.sh"
        sh.write_text(f'#!/bin/bash\n"{venv_python()}" "{here / "server.py"}"\n')
        sh.chmod(0o755)
        launch = str(sh)

    print("\n" + "=" * 62)
    print("세팅 완료!")
    print(f"\n서버 실행:  {launch}")
    print("=" * 62)
