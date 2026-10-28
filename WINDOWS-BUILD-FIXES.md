# Windows build with CUDA - what was wrong and what changed

Building the `p100-optimizations` fork on Windows/MSVC for `sm_60` (Tesla P100). The tree was
written for Linux. Result: a clean Release build, 49 executables in `build/bin/Release`, and
`llama-cli --version` exiting 0.

Source changes: **3 files, +98/-27 lines.** No kernel numerics were touched.

| File | Lines | Issue |
|---|---|---|
| `ggml/src/ggml-cuda/fattn-q4p.cuh` | +3/-2 | `#pragma` between `else` and its statement; nvcc's EDG parser rejects it |
| `ggml/src/ggml-cuda/mmvq-f16.cu` | +57/-25 | MSVC cannot fold a `constexpr` captured from an enclosing lambda |
| `common/common.cpp` | +38/-6 | C11 `aligned_alloc` does not exist on MSVC |

Plus the CUDA runtime DLLs copied into the output folder, and a local build helper,
`win-cuda-build.bat`, that does configure + build + DLL copy.

**The helper is gitignored.** This repository's `.gitignore` excludes `*.bat` at line 4 and `/build*`
at line 46, so a batch script here is invisible to git and would never be committed - the original
name, `build-win.bat`, was caught by both rules. `vc.bat` in the repo root is untracked for the same
reason. Nothing was done to `.gitignore` to work around this, since that would change project-wide
ignore rules for one convenience script; the complete command list is in section 8 instead, so the
build does not depend on the file surviving.

---

## 1. Environment

| Component | Value |
|---|---|
| GPU | 2x Tesla P100-PCIE-16GB (`sm_60`), driver 582.53 |
| CUDA Toolkit | 12.4 (`nvcc` 12.4.131). A `v13.4` directory exists but contains no `nvcc` |
| Compiler | Visual Studio 2022 Community, MSVC 14.43.34808 (`_MSC_VER` 1943) |
| CMake | 4.5.0-rc1, `C:/Program Files/CMake/bin/cmake.exe` (not on `PATH`) |
| Generator | `Visual Studio 17 2022`, `-A x64`, multi-config, `--config Release` |
| Build tree | `build/` |

`_MSC_VER` 1943 is below the 1950 ceiling in CUDA 12.4's `crt/host_config.h`, so the toolchain pair
is supported without patching.

Two things that looked like blockers but were not: the earlier `cfg.log` error *"could not find any
instance of Visual Studio"* did not reproduce (`vswhere` resolves the instance correctly), and no
Linux-only headers blocked the port - `common/console.cpp`, `tools/server/server-common.cpp`,
`tools/mtmd/*` and friends already guard their `unistd.h`/`pthread.h` includes behind `_WIN32`.
The CPU side of the tree compiled untouched.

---

## 2. Defect 1: CUDA toolkit not discoverable by MSBuild

**Symptom.** With `-DGGML_CUDA=ON`, configure failed at `enable_language(CUDA)`:

```
CMake Error at .../CMakeDetermineCUDACompiler.cmake:163 (CMAKE_DETERMINE_COMPILER_ID):
  error : The CUDA Toolkit v12.4 directory '' does not exist.
  Please verify the CUDA Toolkit is installed properly or define the CudaToolkitDir property
```

Before that, `ggml-cuda/CMakeLists.txt:267` reported *"Could not find `nvcc` executable in any
searched paths"* even with an absolute `CMAKE_CUDA_COMPILER`.

**Root cause.** `CUDA 12.4.props` derives `CudaToolkitDir` from three sources, in order: the
`CudaToolkitCustomDir` property, the `CUDA_PATH_V12_4` environment variable, and the registry value
`InstallDir` under `HKLM\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\Cuda\v12.4`. On this
machine the key holds only `64BitInstalled` and `InstalledBy` - there is no `InstallDir` - and no
`CUDA_PATH*` variable is set. All three come up empty, so MSBuild's CUDA build customization gets
`CudaToolkitDir = ''` and every `.cu` target dies. Setting `CMAKE_CUDA_COMPILER` alone is not
enough: the `.vcxproj` still resolves the toolkit independently.

**Fix.** `win-cuda-build.bat` exports the variables for both the configure and the build step, so
the generator and MSBuild agree:

```bat
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_PATH_V12_4=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
```

These are process-scoped. Nothing was written to the registry or to the system environment.

**Result.** `-- CUDA Toolkit found`, `The CUDA compiler identification is NVIDIA 12.4.131 with host
compiler MSVC 19.43.34810.0`, `Using CMAKE_CUDA_ARCHITECTURES=60`, `Including CUDA backend`.

---

## 3. Defect 2: `fattn-q4p.cuh` - `expected a "}"` from the EDG front end

**Symptom.** 18 identical errors in one translation unit, `fattn.cu`:

```
fattn-q4p.cuh(1189): error : expected a "}"
fattn-q4p.cuh(248): note #3196-D: to match this "{"
  detected during instantiation of flash_attn_ext_q4p<1,6,false,true> at line 1248
```

The two reported lines were the kernel's own closing brace and its parameter-list opening brace.
The braces in the body already balanced exactly, 216 open and 216 close, so the message was not
literally true.

**Root cause.** Line 1048 placed a directive between `else` and its controlled statement:

```cpp
        } else
#pragma unroll 2
        for (int p = pg; p < p_end; p += NPG) {
```

GCC's front end, which builds this file on Linux, tolerates it. nvcc's EDG front end uses a
*delayed parse* for template function bodies - the body is stored as tokens and parsed at
instantiation - and that path cannot skip a directive when binding the `else` branch. It consumes
the `for` as the branch, mis-pairs the braces, and reports the mismatch at the function's closing
brace. This explains every misleading aspect of the diagnostic: a line number far from the fault,
a "balanced braces" source file, and errors appearing only during instantiation.

**Fix.** Braced the branch. The pragma still immediately precedes the loop, so the unroll hint is
unchanged:

```cpp
        } else {
#pragma unroll 2
            for (int p = pg; p < p_end; p += NPG) {
            ...
            }
        }
```

`p` is not referenced after the loop, so narrowing its scope is safe. Arithmetic, control flow and
the generated code are unaffected.

**Sweep.** All 76 `.cu`/`.cuh` files were scanned for `} else` followed by a directive: 9 hits, 8 of
them `} else` then `#endif`. `#endif` expands to nothing and those files already compiled, so only
the `#pragma` case needed changing.

**Verification.** `fattn.obj` built, and `ggml-cuda.dll` linked.

---

## 4. Defect 3: `mmvq-f16.cu` - MSVC and constexpr captured from an enclosing lambda

**Symptom.** The device pass succeeded; the host pass (MSVC) rejected the same file. Four separate
rounds, one site at a time, each costing a full compile cycle:

```
error C2672: 'mmvq_f16_32': no matching overloaded function found
note: expression did not evaluate to a constant
note: failure was caused by a read of a variable outside its lifetime
note: see usage of 'this'
fatal error C1003: error count exceeds 100; stopping compilation
```

Affected kernels: `mmvq_f16_32`, `mmvq_f16_s32k`, `mmvq_f16_gen`, `mmvq_f16_q6_K`.

**Root cause.** This file dispatches with nested generic lambdas. An outer lambda derives a
`constexpr` value from a tag parameter, and an inner lambda then uses that value in
template-argument position:

```cpp
auto go = [&](auto q8t, auto ncc) {
    constexpr bool Q8 = decltype(q8t)::value;
    constexpr int  NC = decltype(ncc)::value;
    auto run = [&](auto rpwt, auto nwtt) {
        ...
        mmvq_f16_32<Q8, NC, 2*RPW, NWT, true, MINB><<<...>>>();   // Q8, NC from the enclosing lambda
    };
};
```

`Q8` and `NC` are automatic variables of `go`, so inside `run` they are captured and reached through
the closure object. EDG and GCC constant-fold them and match the overload. MSVC lowers the capture
to `this->...` before its constexpr evaluator runs, sees a read of an object outside its lifetime,
fails to fold the argument, and reports that no overload matches. `this` in the diagnostic is the
lambda closure. This is a known MSVC defect, not a bug in the fork.

**Fix.** Pass the tags down as parameters of the callback instead of capturing them, so nothing in
template-argument position comes from a closure. Derived `constexpr`s move into the inner lambda.

*`mmvq_f16_32`* - tags forwarded through `run`:

```cpp
auto run = [&](auto q8t, auto ncc, auto rpwt, auto nwtt) {
    constexpr bool Q8 = decltype(q8t)::value;
    constexpr int  NC = decltype(ncc)::value;
    ...
};
...
run(q8t, ncc, I2{}, I1{});   // all 9 dispatch sites
```

*`mmvq_f16_s32k`* - `ncc` added as `go`'s first parameter, both call sites updated.

*`mmvq_f16_q6_K`* and *`mmvq_f16_gen`* - the column tag arrives via the existing helper
(`mmvq_f16_nc` / `mmvq_f16_gen_columns`), so forwarding variants were added that also pass the
caller's geometry tags through:

```cpp
// MSVC cannot fold a constexpr captured from an enclosing lambda in template-argument position (it
// reads it through the closure), so extra tags go to the callback as parameters.
template <typename F, typename... T>
static void mmvq_f16_nc_fwd(const int64_t nc, F && f, T &&... t);

template <typename F, typename... T>
static void mmvq_f16_gen_columns_fwd(const int64_t ncols, F && f, T &&... t);
```

The originals are untouched for their other callers.

**This is a mechanical, semantics-preserving transformation.** Every template argument resolves to
exactly the same integer as before; no launch geometry, accumulator or constant changed.

**Sweep.** A static scan of all 76 CUDA files - find any `constexpr <type> <name>` declared in an
enclosing lambda and used inside a `<...>` list in a nested lambda, ignoring names shadowed in the
inner lambda - now reports **zero remaining sites**. An earlier version of the scan under-reported
because its pattern only matched `constexpr bool|int|auto` and missed `constexpr ggml_type T`.

---

## 5. Defect 4: `common.cpp` - `aligned_alloc` is not available on MSVC

**Symptom.** Once the CUDA backend linked, the host library failed:

```
common.cpp(2431): error C3861: 'aligned_alloc': identifier not found
common.cpp(2455): error C3861: 'aligned_alloc': identifier not found
```

**Root cause.** The fork's checkpoint buffer pool allocates 2 MiB-aligned blocks to match huge-page
size, using C11 `aligned_alloc`. MSVC's CRT does not provide it, only `_aligned_malloc`.

The substitution is not a one-for-one rename, for two reasons:

* `_aligned_malloc` must be freed with `_aligned_free`. Mixing it with `free` is undefined
  behaviour, and the CRT debug heap will assert.
* `common_ckpt_alloc` has **two** allocation paths: aligned blocks at or above 4 MiB, and plain
  `malloc` for anything smaller. Both funnel into the single `free(p)` at the end of
  `common_ckpt_free`. Naively replacing that `free` with `_aligned_free` would corrupt the smaller
  blocks.

**Fix.** Three thin wrappers in the file-local anonymous namespace, with the Linux branch left
byte-identical:

```cpp
// Every pointer handed to common_ckpt_free comes from common_ckpt_alloc, so one free covers them
// all. MSVC has no aligned_alloc, and _aligned_malloc cannot mix with malloc/free, so on MSVC the
// small path is aligned too. Linux keeps malloc for small blocks and free for both.
void * ckpt_raw_alloc(size_t n);          // malloc        | _aligned_malloc(n, 64)
void * ckpt_raw_aligned(size_t, size_t);  // aligned_alloc | _aligned_malloc(size, alignment)
void   ckpt_raw_free(void * p);           // free          | _aligned_free
```

On MSVC the small path is aligned as well, which makes one `_aligned_free` correct for every
pointer the pool produces. All five allocation/deallocation sites were routed through the wrappers.

**Safety check first.** `common_ckpt_free` is reached only from `common_ckpt_allocator::deallocate`
(`common.h`) and direct calls, always paired with `common_ckpt_alloc`; nothing external can pass an
unrelated pointer into it.

**Behaviour on Linux is unchanged**, including the `MADV_HUGEPAGE` path, which stays behind
`#if defined(__linux__)`.

---

## 6. Runtime: `--version` crashed with `0xC0000135` after a clean build

**Symptom.** Everything compiled, `build/bin/Release` had 48 executables, but running one printed
nothing and returned `127` under MSYS.

**Root cause.** `0xC0000135`, `STATUS_DLL_NOT_FOUND`. `llama-cli.exe` is a 10 KB shim importing
`llama-cli-impl.dll`, and the dependency chain ends at:

```
ggml-cuda.dll -> cudart64_12.dll, cublas64_12.dll   (from <CUDA>\bin)
```

Those live in the CUDA `bin` directory, which is not on the loader path for a process started from
`build\bin\Release`. Adding it to `PATH` did not help either, since the DLLs must be resolvable
when the process image is mapped.

**Fix.** Copied the CUDA runtime DLLs next to the executables, which is the self-contained layout
this folder layout expects (app directory beats `PATH` in the search order):

| File | Size |
|---|---|
| `cublasLt64_12.dll` | 473.6 MB |
| `cublas64_12.dll` | 100.0 MB |
| `cudart64_12.dll` | 0.6 MB |

`cudart` is statically linked into `ggml-cuda.dll` (`-cudart static`), so the small `cudart64` DLL
is pulled in by cuBLAS rather than by us.

**A caution about how this was diagnosed.** A first recursive `dumpbin /dependents` walk reported
"nothing missing", which sent the investigation down the wrong path for a while. The scan was
broken: it built its set of "available" modules from the app folder *and* the CUDA folder, so it
scored exactly the DLLs that were missing as resolvable. The fix was to stop guessing and read the
actual NTSTATUS, which identified it immediately. Worth remembering for the next silent 127.

---

## 7. Verification

All three binaries, run from `build/bin/Release`, exit code `0x00000000`:

```
llama-cli.exe      version: 0.7.0-dev (build 12323, commit 7461904db)
llama-server.exe   version: 0.7.0-dev (build 12323, commit 7461904db)
llama-serve.exe    version: 0.7.0-dev (build 12323, commit 7461904db)
                   built with MSVC 19.43.34810.0 for x64
```

Commit `7461904db` matches HEAD of `p100-optimizations`. 49 executables total.
`ggml-cuda.dll` is 401.9 MB with `GGML_CUDA_FA_QUANTS=all`, between the 374 MB baseline and the
530 MB size at which `p100-docs/BUILD.md` records long-context `cudaMalloc` failures starting.

`llama-serve.exe` is a copy of `llama-server.exe`, per request.

**Not yet verified: runtime inference.** `--version` does not touch CUDA, so `--list-devices` was
tried as well. It aborts with `CUDA error: out of memory` at `cudaSetDevice`, because another
process already holds 16223 of 16384 MiB on device 1 and 14811 MiB on device 0 - a CUDA context
cannot be created. That process is the user's running server and was deliberately left alone.

The failure originating inside `ggml-cuda.cu` does confirm the backend is compiled in and executing
rather than being a CPU-only fallback, but a real forward pass on Windows is still untested. When
VRAM is free, the remaining check is a short `-m` run, plus the gates in `p100-docs/BUILD.md`
(`tools/gate.sh`: `tg256` around 32 t/s, perplexity 2.6209 +/- 0.0199) to confirm the Pascal tuning
survived the Windows code paths. `--version` reports the commit present at *configure* time and can
lag a rebuild, per the same document.

---

## 8. Reproducing the build

`win-cuda-build.bat` in the repository root does all of the below. CMake is not on `PATH` here, so
the script uses its absolute path. Run it from a plain `cmd` shell, or from an IDE that inherits
`PATH`.

Verified end-to-end after the build was already current: configure succeeds, every target reports
up to date, the three DLLs copy, and no errors are emitted.

```bat
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_PATH_V12_4=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

"C:\Program Files\CMake\bin\cmake.exe" -B build -G "Visual Studio 17 2022" -A x64 ^
  -DGGML_CUDA=ON ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES=60 ^
  -DGGML_CUDA_NCCL=OFF ^
  -DGGML_CUDA_FA_QUANTS=all ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_TESTS=OFF

copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublas64_12.dll"   build\bin\Release\
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublasLt64_12.dll" build\bin\Release\
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll"   build\bin\Release\
```

`LLAMA_CURL=OFF` is needed because OpenSSL is not installed (httplib disables HTTPS). Incremental
rebuilds work; the CUDA runtime copy step is idempotent.

Reasons for the flags, from `p100-docs/BUILD.md`: `sm_60` **alone**, because the Pascal tuning in
`mmvq.cu` is gated on `__CUDA_ARCH_LIST__ == 600` and adding a second architecture silently
disables it; `GGML_CUDA_NCCL=OFF`, since the internal AllReduce is slower on PCIe Pascal; and
`FA_QUANTS=all`, which replaces the deprecated `GGML_CUDA_FA_ALL_QUANTS`. If build time or library
size becomes a problem, upstream's default quant set still covers the serving configuration.

---

## 9. Findings worth keeping

* **`CudaCompile` ignores MSBuild's `UseMultiToolTask`.** The `CMAKE_VS_GLOBALS` settings in
  `CMakeLists.txt` parallelise `cl.exe` but not nvcc, so the ~221 CUDA translation units build
  **serially**. A frozen object count with a live `nvcc`/`cicc`/`ptxas` is normal, not a hang.
* **Fast host-semantics check for a single `.cu`.** Compiling one file with
  `--generate-code=arch=compute_60,code=compute_60` (PTX only) skips the slow `ptxas` step but
  still runs the full MSVC host pass, which is where `C2672` lives. Roughly 9 minutes instead of
  13, and it never disturbs the real build. The final build must still use
  `code=[compute_60,sm_60]`.
* **Isolating a template instantiation is worth the setup.** `fattn-q4p.cuh` was cracked by a
  10-line `.cu` that includes the header and forces `flash_attn_ext_q4p_case<1,6>`, giving a 22
  second reproduction instead of a 25 minute build, and making bisection practical.
* **Bisection needs directive-aware cut points.** Wrapping ranges in `#if 0` is only valid where it
  does not split an `#if`/`#else`/`#endif` region. Two early searches returned confident nonsense
  for exactly that reason - cutting the `#if defined(FLASH_ATTN_AVAILABLE)` line swaps in the
  `NO_DEVICE_CODE` branch, and the error vanishes for a reason that has nothing to do with the
  fault.
* **Idle `MSBuild.exe` processes are node-reuse leftovers, not a live build.** Check for
  `cmake.exe` and `cl`/`nvcc`/`cicc`/`ptxas` rather than counting MSBuild, and `taskkill //F //IM
  MSBuild.exe` before relaunching.

---

## 10. Not done / follow-ups

1. **Run inference on Windows.** Blocked only by the GPUs being full. Once free, confirm a forward
   pass and then the gates, to prove the Windows build matches the Linux numerics.
2. **Upstreamability of the three fixes.** Defect 2 is arguably a latent portability bug upstream
   even though GCC accepts it. Defect 4 is fork-only code. Defect 3 is in fork code. Each is small
   and semantics-preserving, so they may be worth separate discussion with maintainers rather than
   one combined change.
3. **The CUDA toolkit registry entry is still broken on this machine.** `InstallDir` is absent and
   `CUDA_PATH` unset, so any other CUDA project will hit the same wall. Repairing it via the
   installer, or setting the variable system-wide, would remove the need for the environment lines
   in `build-win.bat`.
4. **Keep `win-cuda-build.bat` somewhere durable** if you want it versioned, or add a `.gitignore`
   exception (`!win-cuda-build.bat`). As things stand a fresh clone loses it, and section 8 is the
   only record of the required flags.
5. `unsloth-tmp/` is the nvcc `--keep-dir` (untracked). It holds this build's intermediates and can
   be deleted at any time.
