"""GOF 설치 · 진단.

    python gof_setup.py            # 진단만 (아무것도 안 건드림)
    python gof_setup.py --install  # 받아서 빌드까지

GOF 는 CUDA 확장 세 개를 직접 컴파일해야 한다. 그래서 pip install 한 줄로 안 끝난다.

    diff-gaussian-rasterization   불투명도장 래스터라이저   nvcc 필요
    simple-knn                    이웃 탐색                 nvcc 필요
    tetra-triangulation           사면체 분할 (메쉬 추출)   nvcc + CMake + CGAL

⚠ 윈도우에서 막히는 지점은 대체로 세 번째다.
  tetra-triangulation 은 CGAL 헤더를 찾는 CMakeLists 가 리눅스 기준으로 쓰여 있다.
  앞의 둘이 빌드돼도 이것만 실패하면 학습(train.py)은 되고 메쉬 추출만 안 된다.
  그 경우 현실적인 우회는 WSL2 다 — GPU 패스스루로 같은 3070 을 그대로 쓴다.
  이 스크립트는 어디까지 됐는지 단계별로 정확히 알려준다.
"""
import os, sys, json, shutil, subprocess, argparse

# 콘솔이 cp949 면 '—' 같은 문자에서 죽는다. .bat 는 chcp 65001 을 걸어주지만
# 직접 실행할 때를 대비해 여기서도 한 번 더 막는다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


REPO = "https://github.com/autonomousvision/gaussian-opacity-fields.git"
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "gaussian-opacity-fields")
SUBMODULES = [
    ("diff-gaussian-rasterization", "submodules/diff-gaussian-rasterization"),
    ("simple-knn", "submodules/simple-knn"),
    ("tetra-triangulation", "submodules/tetra-triangulation"),
]


def sh(cmd, cwd=None, quiet=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", shell=isinstance(cmd, str))
    if not quiet and p.returncode != 0:
        out = ((p.stdout or "") + (p.stderr or "")).strip().split("\n")
        for l in out[-12:]:
            print("    | " + l[:200])
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def which(name):
    return shutil.which(name)


def diagnose():
    print("=" * 62)
    print("  GOF 환경 진단")
    print("=" * 62)
    st = {}

    # GPU
    rc, out = sh(["nvidia-smi", "--query-gpu=name,memory.total",
                  "--format=csv,noheader"], quiet=True)
    st["gpu"] = out.strip().split("\n")[0] if rc == 0 else None
    print(f"  GPU        {st['gpu'] or '못 찾음'}")

    # nvcc — torch 의 cu124 런타임만으로는 확장을 못 만든다. 툴킷이 따로 필요하다.
    nvcc = which("nvcc")
    if nvcc:
        rc, out = sh([nvcc, "--version"], quiet=True)
        ver = ""
        for l in out.split("\n"):
            if "release" in l:
                ver = l.split("release")[-1].strip()
        st["nvcc"] = ver or nvcc
        print(f"  nvcc       {st['nvcc']}")
    else:
        st["nvcc"] = None
        print("  nvcc       [없음] ← CUDA Toolkit 을 설치해야 합니다")

    # MSVC
    msvc_root = r"C:\Program Files\Microsoft Visual Studio\2022"
    vers = []
    for ed in ("Community", "Professional", "Enterprise", "BuildTools"):
        d = os.path.join(msvc_root, ed, "VC", "Tools", "MSVC")
        if os.path.isdir(d):
            vers += [f"{ed}/{v}" for v in os.listdir(d)]
    st["msvc"] = vers
    print(f"  MSVC       {', '.join(vers) if vers else '[없음]'}")

    # torch
    try:
        import torch
        st["torch"] = f"{torch.__version__} (cuda {torch.version.cuda}, " \
                      f"사용가능 {torch.cuda.is_available()})"
    except Exception as e:
        st["torch"] = f"[없음] {e}"
    print(f"  torch      {st['torch']}")

    for t in ("git", "cmake"):
        st[t] = which(t)
        print(f"  {t:10s} {st[t] or '[없음]'}")

    # GOF
    gof = os.environ.get("GOF_DIR", "") or DEFAULT_DIR
    st["gof_dir"] = gof if os.path.exists(os.path.join(gof, "train.py")) else None
    print(f"  GOF        {st['gof_dir'] or '[미설치]'}")

    if st["gof_dir"]:
        for name, _ in SUBMODULES:
            mod = name.replace("-", "_")
            rc, _ = sh([sys.executable, "-c", f"import {mod}"], quiet=True)
            print(f"    {name:30s} {'설치됨' if rc == 0 else '[빌드 안 됨]'}")
            st[name] = (rc == 0)

    # 결론
    print("-" * 62)
    blockers = []
    if not st["nvcc"]:
        blockers.append(
            "CUDA Toolkit 이 없습니다. torch 의 cu124 는 '런타임'이라 컴파일러(nvcc)가\n"
            "    안 들어 있습니다. 확장을 빌드하려면 툴킷을 따로 받아야 합니다.\n"
            "    → https://developer.nvidia.com/cuda-downloads (약 3GB)\n"
            "    ⚠ 12.4 를 받으면 MSVC 14.43 을 '지원 안 함'으로 거부합니다 (12.4 상한 14.39).\n"
            "      12.6 이상을 받으세요.")
    if not st["msvc"]:
        blockers.append("Visual Studio 2022 의 'C++를 사용한 데스크톱 개발' 워크로드가 필요합니다.")
    if not st["cmake"]:
        blockers.append("CMake 가 필요합니다 (tetra-triangulation 빌드용). winget install Kitware.CMake")

    if blockers:
        print("  막힌 것:")
        for b in blockers:
            print(f"  - {b}")
    else:
        print("  빌드 전제조건은 모두 갖춰졌습니다.")
    print("=" * 62)
    return st


def install(gof_dir, py):
    if not which("git"):
        print("[!] git 이 없습니다.")
        return 1
    if not os.path.exists(os.path.join(gof_dir, "train.py")):
        print(f"[1/2] 내려받기 → {gof_dir}")
        rc, _ = sh(["git", "clone", "--recursive", REPO, gof_dir])
        if rc != 0:
            print("[!] clone 실패")
            return 1
    else:
        print(f"[1/2] 이미 있습니다 → {gof_dir}")

    print(f"[2/2] CUDA 확장 빌드 ({py})")
    fails = []
    for name, rel in SUBMODULES:
        d = os.path.join(gof_dir, *rel.split("/"))
        if not os.path.isdir(d):
            print(f"  {name:30s} [폴더 없음 — --recursive 로 다시 clone 하세요]")
            fails.append(name)
            continue
        print(f"  {name} 빌드 중...")
        rc, _ = sh([py, "-m", "pip", "install", "-e", d])
        print(f"  {name:30s} {'OK' if rc == 0 else '실패'}")
        if rc != 0:
            fails.append(name)

    print("-" * 62)
    if not fails:
        print("  전부 빌드됐습니다.")
        print(f"  환경변수를 걸어두세요:  GOF_DIR={gof_dir}")
        return 0
    print(f"  실패: {', '.join(fails)}")
    if fails == ["tetra-triangulation"]:
        print("  학습(train.py)은 되지만 메쉬 추출이 안 됩니다.")
        print("  CGAL 이 원인이면 WSL2 로 옮기는 게 가장 빠릅니다.")
    return 1


def main():
    ap = argparse.ArgumentParser(description="GOF 설치·진단")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--dir", default=os.environ.get("GOF_DIR", "") or DEFAULT_DIR)
    ap.add_argument("--python", default=sys.executable,
                    help="확장을 설치할 파이썬 (기본: 지금 인터프리터)")
    a = ap.parse_args()

    st = diagnose()
    if not a.install:
        print("\n실제로 받으려면: python gof_setup.py --install")
        return 0
    if not st["nvcc"]:
        print("\n[중단] nvcc 없이는 빌드가 무조건 실패합니다. 위 안내를 먼저 처리하세요.")
        return 1
    return install(a.dir, a.python)


if __name__ == "__main__":
    sys.exit(main())
