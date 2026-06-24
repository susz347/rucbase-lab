"""
MSVC build script for NSG project - no CMake required.
Usage: python build_msvc.py [clean|build|rebuild]
"""
import os
import sys
import subprocess
import glob

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build_msvc")

# MSVC flags
CFLAGS = [
    "/nologo", "/O2", "/arch:AVX2", "/openmp",
    "/std:c++17", "/EHsc", "/DINFO", "/DNDEBUG",
    f"/I{INCLUDE_DIR}",
]

# Source files
LIB_SOURCES = [
    os.path.join(SRC_DIR, "index.cpp"),
    os.path.join(SRC_DIR, "index_nsg.cpp"),
]

TEST_SOURCES = {
    "test_nsg_index": os.path.join(TESTS_DIR, "test_nsg_index.cpp"),
    "test_nsg_search": os.path.join(TESTS_DIR, "test_nsg_search.cpp"),
    "test_nsg_optimized_search": os.path.join(TESTS_DIR, "test_nsg_optimized_search.cpp"),
}


def find_cl():
    """Find MSVC compiler."""
    # Try direct path
    cl_path = r"E:\visutal studio c++\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe"
    if os.path.exists(cl_path):
        return cl_path

    # Try to find via 'where'
    try:
        result = subprocess.run(["where", "cl"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "cl"


def run_cmd(cmd, desc=""):
    """Run a command and print output."""
    print(f"[{desc}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(1)
    return result.returncode == 0


def compile_objects(cl_exe, sources, extra_flags=None):
    """Compile .cpp files to .obj files."""
    flags = CFLAGS.copy()
    if extra_flags:
        flags.extend(extra_flags)

    obj_files = []
    for src in sources:
        obj = os.path.join(BUILD_DIR, os.path.basename(src).replace(".cpp", ".obj"))
        obj_files.append(obj)

        # Check if recompilation is needed
        if os.path.exists(obj) and os.path.getmtime(obj) > os.path.getmtime(src):
            print(f"  [SKIP] {os.path.basename(src)} (up to date)")
            continue

        cmd = [cl_exe, "/c", src, f"/Fo{obj}"] + flags
        print(f"  [COMPILE] {os.path.basename(src)}")
        result = subprocess.run(" ".join(cmd), cwd=BUILD_DIR, shell=True)
        if result.returncode != 0:
            print(f"ERROR: Compilation failed for {src}")
            sys.exit(1)

    return obj_files


def build():
    """Build the NSG library and test executables."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    cl_exe = find_cl()

    # Check if compiler is accessible
    try:
        result = subprocess.run(f'"{cl_exe}" 2>&1', capture_output=True, text=True, shell=True)
    except Exception as e:
        print(f"ERROR: Cannot run MSVC compiler: {e}")
        print("Make sure to run this script from a Developer Command Prompt for VS.")
        sys.exit(1)

    print(f"Using compiler: {cl_exe}")
    print(f"Building in: {BUILD_DIR}")

    # Compile library objects
    print("\n--- Compiling library sources ---")
    lib_objs = compile_objects(cl_exe, LIB_SOURCES)

    # Compile test executables
    print("\n--- Compiling test executables ---")
    for name, src in TEST_SOURCES.items():
        if not os.path.exists(src):
            print(f"  [SKIP] {name} (source not found)")
            continue

        obj = os.path.join(BUILD_DIR, os.path.basename(src).replace(".cpp", ".obj"))
        cmd = [cl_exe, "/c", src, f"/Fo{obj}"] + CFLAGS + ["/Fe:" + os.path.join(BUILD_DIR, name + ".exe")]
        # First compile test source
        test_cmd = [cl_exe, "/c", src, f"/Fo{obj}"] + CFLAGS
        result = subprocess.run(" ".join(test_cmd), cwd=BUILD_DIR, shell=True)

        # Link with library objects
        all_objs = lib_objs + [obj]
        link_cmd = [cl_exe] + all_objs + [
            f"/Fe{os.path.join(BUILD_DIR, name)}.exe",
            "/openmp", "/EHsc"
        ]
        print(f"  [LINK] {name}.exe")
        result = subprocess.run(" ".join(link_cmd), cwd=BUILD_DIR, shell=True)
        if result.returncode != 0:
            print(f"ERROR: Link failed for {name}")
            sys.exit(1)

    print("\n--- Build complete ---")
    for name in TEST_SOURCES:
        exe = os.path.join(BUILD_DIR, name + ".exe")
        if os.path.exists(exe):
            size_mb = os.path.getsize(exe) / (1024 * 1024)
            print(f"  {name}.exe ({size_mb:.1f} MB)")


def clean():
    """Clean build artifacts."""
    if os.path.exists(BUILD_DIR):
        import shutil
        shutil.rmtree(BUILD_DIR)
        print(f"Removed {BUILD_DIR}")
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "clean":
        clean()
    elif cmd == "rebuild":
        clean()
        build()
    else:
        build()
