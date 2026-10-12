# unholy, built with the TheRock SDK clang (wrapped for NixOS headers by therock-sdk-nix) as both the
# C++ and the HIP compiler, as a fat binary for the R9700s and the Strix Halo iGPU.
{
  lib,
  stdenv,
  cmake,
  ninja,
  patchelf,
  pkg-config,
  pcre2,
  httplib,
  rocmSdk,
  rocmSdkCc,
  gpuTargets,
}:

stdenv.mkDerivation {
  pname = "unholy";
  version = "0.0.1";

  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      let
        base = baseNameOf path;
      in
      !(lib.hasPrefix "build" base) && base != "result" && !(lib.hasSuffix ".nix" base);
  };

  nativeBuildInputs = [
    cmake
    ninja
    patchelf
    pkg-config
  ];

  buildInputs = [
    pcre2
    httplib
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "CMAKE_CXX_COMPILER" "${rocmSdkCc}/bin/clang++")
    (lib.cmakeFeature "CMAKE_HIP_COMPILER" "${rocmSdkCc}/bin/clang++")
    (lib.cmakeFeature "CMAKE_HIP_COMPILER_ROCM_ROOT" "${rocmSdk}")
    (lib.cmakeFeature "CMAKE_PREFIX_PATH" "${rocmSdk}")
    (lib.cmakeFeature "CMAKE_HIP_ARCHITECTURES" (lib.concatStringsSep ";" gpuTargets))
  ];

  env = {
    ROCM_PATH = "${rocmSdk}";
    HIP_DEVICE_LIB_PATH = "${rocmSdk}/lib/llvm/amdgcn/bitcode";
  };

  # Linked by the SDK clang, not nix's ld wrapper: add the SDK and our own lib dir to the runpath.
  postFixup = ''
    for f in $out/bin/*; do
      [ -L "$f" ] || patchelf --add-rpath ${rocmSdk}/lib:${lib.getLib pcre2}/lib:$out/lib "$f"
    done
  '';

  passthru = {
    inherit rocmSdk rocmSdkCc gpuTargets;
  };

  meta = {
    description = "Vertical DeepSeek-V4-Flash serving engine for 2x R9700 + Strix Halo (ROCm ${rocmSdk.version})";
    platforms = [ "x86_64-linux" ];
    license = lib.licenses.asl20;
    mainProgram = "unholy";
  };
}
