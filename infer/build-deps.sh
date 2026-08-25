#!/usr/bin/env bash
# build-deps.sh -- everything stage four (README §20) needs, from nothing
# to a running hello_inference: our Dawn as a monolithic shared library,
# an onnxruntime built against that same Dawn source tree with
# --use_external_dawn, and this repo configured against both.
#
#   ./infer/build-deps.sh                 # all steps, idempotent
#   ./infer/build-deps.sh audit           # only re-check Dawn's DEPS
#   ./infer/build-deps.sh dawn ort app    # skip the source steps
#
# The order is not the build order: the Dawn revision is READ OUT of
# onnxruntime's cmake/deps.txt, so ORT is cloned first although Dawn is
# built first. Dawn and ORT must come from the same Dawn tree -- see the
# revision-lock discussion in README §20.
#
# Overridable:
#   ORT_TAG   onnxruntime tag to build (its deps.txt picks the Dawn tag)
#   DAWN_TAG  override the Dawn tag instead of reading it from ORT
#   ORT_SRC   DAWN_SRC     source trees        (~/self-builds/*)
#   ORT_ROOT  DAWN_ROOT    install prefixes    (~/opt/*)
#   ORT_JOBS  DAWN_JOBS    build parallelism
set -euo pipefail

ORT_TAG=${ORT_TAG:-v1.29.0}
ORT_SRC=${ORT_SRC:-$HOME/self-builds/onnxruntime}
DAWN_SRC=${DAWN_SRC:-$HOME/self-builds/dawn}
ORT_ROOT=${ORT_ROOT:-$HOME/opt/onnxruntime-webgpu-extdawn}
DAWN_ROOT=${DAWN_ROOT:-$HOME/opt/dawn}
ORT_JOBS=${ORT_JOBS:-4}
DAWN_JOBS=${DAWN_JOBS:-$(nproc)}
REPO=$(cd "$(dirname "$0")/.." && pwd)

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# clone if absent, otherwise fetch; then detach at $2
checkout() { # <dir> <ref> <url>
    if [ ! -d "$1/.git" ]; then
        mkdir -p "$(dirname "$1")"
        git clone "$3" "$1"          # full clone: the tag checkout and the
    else                             # DEPS audit both need history
        git -C "$1" fetch --tags origin
    fi
    git -C "$1" checkout "$2"
}

# the Dawn revision ORT pins for $ORT_TAG (the revision lock)
dawn_tag() {
    [ -n "${DAWN_TAG:-}" ] && { echo "$DAWN_TAG"; return; }
    [ -d "$ORT_SRC/.git" ] || die "no ORT tree at $ORT_SRC; run: $0 ort-src"
    git -C "$ORT_SRC" show "$ORT_TAG:cmake/deps.txt" |
        sed -n 's#.*dawn/archive/refs/tags/\(v[0-9.]*\)\.zip.*#\1#p' | head -1
}

step_ort_src() {
    log "onnxruntime $ORT_TAG -> $ORT_SRC"
    checkout "$ORT_SRC" "$ORT_TAG" https://github.com/microsoft/onnxruntime.git
    # build.py inits cmake/external/onnx and friends itself, but it needs
    # its own Python packages present. Only touch a venv if the system
    # interpreter is missing them.
    if ! python3 -c 'import flatbuffers, numpy, packaging, google.protobuf' 2>/dev/null; then
        log "installing ORT's Python build deps into $ORT_SRC/.venv"
        [ -x "$ORT_SRC/.venv/bin/python" ] || python3 -m venv "$ORT_SRC/.venv"
        "$ORT_SRC/.venv/bin/python" -m pip install -q --upgrade pip setuptools wheel
        "$ORT_SRC/.venv/bin/python" -m pip install -q -r "$ORT_SRC/requirements.txt"
    fi
    echo "Dawn pin for $ORT_TAG: $(dawn_tag)"
}

step_dawn_src() {
    local tag; tag=$(dawn_tag)
    [ -n "$tag" ] || die "could not read the Dawn pin from $ORT_TAG's cmake/deps.txt"
    log "Dawn $tag -> $DAWN_SRC"
    checkout "$DAWN_SRC" "$tag" https://dawn.googlesource.com/dawn
    (cd "$DAWN_SRC" && python3 tools/fetch_dawn_dependencies.py)
    step_audit
}

# fetch_dawn_dependencies.py updates existing shallow clones with
# `git fetch origin <sha> --depth 1`, which the googlesource mirrors
# refuse for commits older than the clone's tip: it prints "Checking out
# tag ..." and leaves the dependency where it was. Stale deps produce a
# Dawn whose generated headers do not match ORT's, and that only fails
# much later, in the ORT build. So: verify every checked-out third_party
# against DEPS, and repair by fetching the pinned commit directly (from
# GitHub when the mirror refuses).
step_audit() {
    log "auditing $DAWN_SRC/third_party against DEPS"
    (cd "$DAWN_SRC" && python3 - <<'PY'
import os, subprocess, sys

scope = {}
exec(compile(open('DEPS').read(), 'DEPS', 'exec'),
     {'Var': lambda k: '{%s}' % k, 'Str': str}, scope)
dvars = {k: v for k, v in scope.get('vars', {}).items() if isinstance(v, str)}

def expand(url):
    for k, v in dvars.items():
        url = url.replace('{%s}' % k, v)
    return url

def run(*cmd):
    return subprocess.call(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0

bad = []
for path, dep in scope['deps'].items():
    url = dep.get('url') if isinstance(dep, dict) else dep
    if not url or '@' not in url:
        continue                                  # CIPD package, not a checkout
    url, sha = expand(url).rsplit('@', 1)
    if not os.path.isdir(os.path.join(path, '.git')):
        continue                                  # condition was false; not fetched
    have = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD']).decode().strip()
    if have == sha:
        continue
    print('STALE %s\n      have %s\n      want %s' % (path, have, sha))
    # the mirror first, then GitHub for the external/github.com/* mirrors
    urls = [url]
    marker = 'external/github.com/'
    if marker in url:
        urls.append('https://github.com/' + url.split(marker, 1)[1])
    for u in urls:
        if run('git', '-C', path, 'fetch', '--depth', '1', u, sha) and \
           run('git', '-C', path, 'checkout', '--detach', sha):
            print('      repaired from %s' % u)
            break
    else:
        bad.append(path)

if bad:
    print('\ncould not repair: %s' % ' '.join(bad), file=sys.stderr)
    print('fix by hand, then re-run: infer/build-deps.sh audit', file=sys.stderr)
    sys.exit(1)
print('all pinned dependencies match DEPS')
PY
    )
}

step_dawn() {
    log "building Dawn -> $DAWN_ROOT"
    rm -rf "$DAWN_SRC/out/Release" "$DAWN_ROOT"
    cmake -S "$DAWN_SRC" -B "$DAWN_SRC/out/Release" -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
          -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED -DDAWN_ENABLE_INSTALL=ON -DDAWN_FETCH_DEPENDENCIES=OFF \
          -DDAWN_ENABLE_VULKAN=ON -DDAWN_USE_WAYLAND=ON -DDAWN_USE_X11=OFF -DDAWN_ENABLE_RTTI=ON \
          -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF -DTINT_BUILD_TESTS=OFF \
          -DTINT_BUILD_CMD_TOOLS=OFF -DTINT_BUILD_SPV_READER=OFF \
          -DCMAKE_INSTALL_PREFIX="$DAWN_ROOT"
    cmake --build "$DAWN_SRC/out/Release" -j "$DAWN_JOBS"
    cmake --install "$DAWN_SRC/out/Release"
}

step_ort() {
    [ -d "$DAWN_SRC/.git" ] || die "no Dawn tree at $DAWN_SRC; run: $0 dawn-src"
    log "building onnxruntime (external Dawn from $DAWN_SRC) -> $ORT_ROOT"
    rm -rf "$ORT_SRC/build-webgpu-extdawn" "$ORT_ROOT"
    [ -x "$ORT_SRC/.venv/bin/python" ] && PATH="$ORT_SRC/.venv/bin:$PATH"
    (cd "$ORT_SRC" && CC=clang CXX=clang++ ./build.sh \
        --config Release --build_dir build-webgpu-extdawn \
        --build_shared_lib --use_webgpu --use_external_dawn --parallel "$ORT_JOBS" --skip_tests \
        --compile_no_warning_as_error --cmake_generator Ninja \
        --cmake_extra_defines "onnxruntime_CUSTOM_DAWN_SRC_PATH=$DAWN_SRC" \
                              onnxruntime_BUILD_UNIT_TESTS=OFF \
                              "CMAKE_INSTALL_PREFIX=$ORT_ROOT")
    cmake --install "$ORT_SRC/build-webgpu-extdawn/Release"

    # Two Dawns in one process cannot share a device: ORT must link no
    # Dawn implementation and export no wgpu symbols that could interpose
    # on ours.
    local lib; lib=$(ls "$ORT_ROOT"/lib64/libonnxruntime.so "$ORT_ROOT"/lib/libonnxruntime.so 2>/dev/null | head -1)
    [ -n "$lib" ] || die "no libonnxruntime.so under $ORT_ROOT"
    if readelf -d "$lib" | grep NEEDED | grep -q webgpu_dawn; then
        die "$lib links its own Dawn -- --use_external_dawn did not take"
    fi
    if [ "$(nm -D --defined-only "$lib" | grep -c ' wgpu')" != 0 ]; then
        die "$lib exports wgpu symbols -- they would interpose on our Dawn"
    fi
    echo "ok: $lib consumes an external Dawn and exports no wgpu symbols"
}

step_app() {
    log "configuring $REPO against ORT_ROOT=$ORT_ROOT DAWN_ROOT=$DAWN_ROOT"
    cmake -S "$REPO" -B "$REPO/build" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DORT_ROOT="$ORT_ROOT" -DDAWN_ROOT="$DAWN_ROOT"
    cmake --build "$REPO/build" --clean-first --target hello_inference
    ctest --test-dir "$REPO/build" -R infer_ --output-on-failure
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(ort-src dawn-src dawn ort app)
for s in "${steps[@]}"; do
    case $s in
        ort-src)  step_ort_src ;;
        dawn-src) step_dawn_src ;;
        audit)    step_audit ;;
        dawn)     step_dawn ;;
        ort)      step_ort ;;
        app)      step_app ;;
        *) die "unknown step '$s' (ort-src dawn-src audit dawn ort app)" ;;
    esac
done
log "done"
