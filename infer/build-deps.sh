#!/usr/bin/env bash
# build-deps.sh -- everything stage four (README §20) needs, from nothing
# to a running hello_inference: our Dawn as a monolithic shared library,
# an onnxruntime built against that same Dawn source tree with
# --use_external_dawn, and this repo configured against both.
#
#   ./infer/build-deps.sh                 # all steps, idempotent
#   ./infer/build-deps.sh audit           # only re-check Dawn's DEPS
#   ./infer/build-deps.sh dawn ort app    # skip the source steps
#   CLEAN=1 ./infer/build-deps.sh dawn    # from scratch, not incrementally
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
#   CC        CXX          compilers (default: whichever clang is installed)
#   DAWN_PICKS upstream Dawn commits to apply on top of the pinned tag
#   DAWN_PATCH_DIR  local patches (infer/patches/dawn/*.patch), applied after
#   CLEAN=1   wipe build dirs and install prefixes first. Otherwise the build
#             steps are incremental, and only start over by themselves when
#             the source revision they last built from has changed.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ORT_TAG=${ORT_TAG:-v1.29.0}
ORT_SRC=${ORT_SRC:-$HOME/self-builds/onnxruntime}
DAWN_SRC=${DAWN_SRC:-$HOME/self-builds/dawn}
ORT_ROOT=${ORT_ROOT:-$HOME/opt/onnxruntime-webgpu-extdawn}
DAWN_ROOT=${DAWN_ROOT:-$HOME/opt/dawn}
ORT_JOBS=${ORT_JOBS:-4}
DAWN_JOBS=${DAWN_JOBS:-$(nproc)}
# Upstream fixes that landed after the tag ORT pins, applied to the working
# tree as unmodified cherry-picks. Each entry must be a commit already on
# Dawn's main branch, never a local invention, so the list empties itself
# when ORT's pin next moves forward. Space-separated shas; empty is normal.
#
# Deliberately NOT carried here, after testing (2026-08-25):
#   2f24fba5fd  "[dawn] Fallback to host memory for dmabuf import in
#               SharedTextureMemory". Lets the import get past memory-type
#               selection, but the allocation that follows is refused by the
#               NVIDIA proprietary driver (VK_ERROR_OUT_OF_DEVICE_MEMORY:
#               Dawn's dma-buf path passes no VkMemoryDedicatedAllocateInfo,
#               and the type it picks is not the one that imports). The
#               driver closes the fd on that failure and Dawn's error path
#               closes it again, so the process aborts instead of reporting
#               a failed edge. Without the pick, dmabuf -> webgpu fails
#               cleanly and the rest of the matrix still runs. See README,
#               "Cherry-picks onto the pinned Dawn".
DAWN_PICKS=${DAWN_PICKS:-}
# Local patches, for what upstream does not have yet. Each file in the
# directory is a `git diff` against the pinned tag, applied after DAWN_PICKS
# with the same idempotency rules. Every patch here is a candidate for an
# upstream change and should say so in its header; when it lands, move its
# sha to DAWN_PICKS and delete the file.
DAWN_PATCH_DIR=${DAWN_PATCH_DIR:-$REPO/infer/patches/dawn}

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Dawn and ORT must be built by the same compiler, and it has to be clang:
# ORT's WebGPU EP and Dawn share C++ types across the library boundary, so
# the two trees have to agree on the toolchain. Distros disagree on where
# clang lives -- Debian/Ubuntu ship only clang-N and /usr/lib/llvm-N/bin --
# so find it rather than assuming an unsuffixed name is on PATH.
find_clang() { # <clang|clang++> -> path, empty if absent
    local c
    c=$(command -v "$1" || true)
    [ -n "$c" ] || c=$(ls /usr/bin/"$1"-[0-9]* /usr/lib/llvm-*/bin/"$1" \
                       2>/dev/null | sort -V | tail -1)
    echo "$c"
}
CC=${CC:-$(find_clang clang)}
CXX=${CXX:-$(find_clang clang++)}
[ -n "$CC" ] && [ -n "$CXX" ] ||
    die "no clang found -- install one (Debian: apt install clang) or set CC= and CXX="

# Incremental unless told otherwise. A build directory remembers the
# revision it was configured from in a stamp file; if that changed (the
# pin moved, a different tag was asked for), the old objects are not to be
# trusted -- Dawn's generated headers and proc table differ between
# revisions -- and the build starts over. Patches applied with git apply
# do not move HEAD, but they touch files, and ninja sees that on its own.
CLEAN=${CLEAN:-}
stamp_of() { cat "$1/.build-deps-stamp" 2>/dev/null; }
needs_clean() { # <build-dir> <expected stamp>
    [ -n "$CLEAN" ] && { echo "CLEAN=1"; return 0; }
    [ -f "$1/build.ninja" ] || { echo "no previous build"; return 0; }
    [ "$(stamp_of "$1")" = "$2" ] || { echo "built from a different revision"; return 0; }
    return 1
}

# clone if absent, otherwise fetch; then detach at $2
checkout() { # <dir> <ref> <url>
    if [ ! -d "$1/.git" ]; then
        mkdir -p "$(dirname "$1")"
        git clone "$3" "$1"          # full clone: the tag checkout and the
    else                             # DEPS audit both need history
        git -C "$1" remote set-url origin "$3"   # repair trees cloned from
        git -C "$1" fetch --tags origin          # a tagless URL (see below)
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
    # From the GitHub mirror, not dawn.googlesource.com: the upstream repo
    # publishes no tags at all, and the vYYYYMMDD.HHMMSS releases ORT pins
    # exist only as GitHub refs. Same history, so an existing googlesource
    # clone is repaired by checkout()'s remote set-url rather than re-cloned.
    checkout "$DAWN_SRC" "$tag" https://github.com/google/dawn.git
    step_picks
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

# Apply one patch (on stdin) to the Dawn tree, idempotently: a patch that
# reverse-applies cleanly is already in, so it is skipped rather than
# applied twice. $1 names it for the log.
apply_patch() { # <name>  (patch on stdin)
    local patch; patch=$(cat)
    if printf '%s\n' "$patch" | git -C "$DAWN_SRC" apply --reverse --check - 2>/dev/null; then
        echo "  $1: already applied"
        return 0
    fi
    printf '%s\n' "$patch" | git -C "$DAWN_SRC" apply - ||
        die "could not apply $1; the pin may have moved past it"
    echo "  $1: applied"
}

# DAWN_PICKS: upstream commits cherry-picked onto the checked-out tag as
# working-tree patches (no commit identity needed). One already in the
# tag's history is skipped outright. Then DAWN_PATCH_DIR: local patches.
step_picks() {
    if [ -n "${DAWN_PICKS:-}" ]; then
        log "applying upstream Dawn fixes on top of $(dawn_tag)"
        for sha in $DAWN_PICKS; do
            if git -C "$DAWN_SRC" merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
                echo "  ${sha:0:10}: already in the pinned tag"
                continue
            fi
            git -C "$DAWN_SRC" show "$sha" | apply_patch "${sha:0:10}"
        done
    fi
    if ls "$DAWN_PATCH_DIR"/*.patch >/dev/null 2>&1; then
        log "applying local Dawn patches from $DAWN_PATCH_DIR"
        for f in "$DAWN_PATCH_DIR"/*.patch; do
            apply_patch "$(basename "$f")" < "$f"
        done
    fi
}

step_dawn() {
    local build="$DAWN_SRC/out/Release" stamp why
    stamp=$(git -C "$DAWN_SRC" rev-parse HEAD)
    if why=$(needs_clean "$build" "$stamp"); then
        log "building Dawn from scratch ($why) -> $DAWN_ROOT"
        rm -rf "$build" "$DAWN_ROOT"
    else
        log "building Dawn incrementally -> $DAWN_ROOT"
    fi
    # re-running configure on an existing build dir is cheap and picks up
    # any flag edited below without a CLEAN=1
    cmake -S "$DAWN_SRC" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
          -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED -DDAWN_ENABLE_INSTALL=ON -DDAWN_FETCH_DEPENDENCIES=OFF \
          -DDAWN_ENABLE_VULKAN=ON -DDAWN_USE_WAYLAND=ON -DDAWN_USE_X11=OFF -DDAWN_ENABLE_RTTI=ON \
          -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF -DTINT_BUILD_TESTS=OFF \
          -DTINT_BUILD_CMD_TOOLS=OFF -DTINT_BUILD_SPV_READER=OFF \
          -DCMAKE_INSTALL_PREFIX="$DAWN_ROOT"
    cmake --build "$build" -j "$DAWN_JOBS"
    cmake --install "$build"
    echo "$stamp" > "$build/.build-deps-stamp"
}

step_ort() {
    [ -d "$DAWN_SRC/.git" ] || die "no Dawn tree at $DAWN_SRC; run: $0 dawn-src"
    # ORT compiles against the Dawn tree, so its objects are tied to both
    # revisions
    local build="$ORT_SRC/build-webgpu-extdawn/Release" stamp why
    stamp="ort=$(git -C "$ORT_SRC" rev-parse HEAD) dawn=$(git -C "$DAWN_SRC" rev-parse HEAD)"
    if why=$(needs_clean "$build" "$stamp"); then
        log "building onnxruntime from scratch ($why; external Dawn from $DAWN_SRC) -> $ORT_ROOT"
        rm -rf "$ORT_SRC/build-webgpu-extdawn" "$ORT_ROOT"
    else
        log "building onnxruntime incrementally -> $ORT_ROOT"
    fi
    [ -x "$ORT_SRC/.venv/bin/python" ] && PATH="$ORT_SRC/.venv/bin:$PATH"
    (cd "$ORT_SRC" && CC="$CC" CXX="$CXX" ./build.sh \
        --config Release --build_dir build-webgpu-extdawn \
        --build_shared_lib --use_webgpu --use_external_dawn --parallel "$ORT_JOBS" --skip_tests \
        --compile_no_warning_as_error --cmake_generator Ninja \
        --cmake_extra_defines "onnxruntime_CUSTOM_DAWN_SRC_PATH=$DAWN_SRC" \
                              onnxruntime_BUILD_UNIT_TESTS=OFF \
                              "CMAKE_INSTALL_PREFIX=$ORT_ROOT")
    cmake --install "$build"
    echo "$stamp" > "$build/.build-deps-stamp"

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
    cmake --build "$REPO/build" ${CLEAN:+--clean-first} --target hello_inference
    ctest --test-dir "$REPO/build" -R infer_ --output-on-failure
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(ort-src dawn-src dawn ort app)
for s in "${steps[@]}"; do
    case $s in
        ort-src)  step_ort_src ;;
        dawn-src) step_dawn_src ;;
        audit)    step_audit ;;
        picks)    step_picks ;;
        dawn)     step_dawn ;;
        ort)      step_ort ;;
        app)      step_app ;;
        *) die "unknown step '$s' (ort-src dawn-src audit picks dawn ort app)" ;;
    esac
done
log "done"
