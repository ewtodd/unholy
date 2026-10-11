{
  description = "unholy: a vertical DeepSeek-V4-Flash serving engine for 2x R9700 (gfx1201) + Strix Halo (gfx1151)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    therock-sdk = {
      url = "github:ewtodd/therock-sdk-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      therock-sdk,
    }:
    let
      system = "x86_64-linux";
      # Callers (the NixOS host) pass their own pkgs through `lib.mkUnholy` so there is one nixpkgs;
      # the standalone instance below allows the unfree AMD SDK.
      mkUnholy =
        {
          pkgs,
          gpuTargets ? [
            "gfx1201"
            "gfx1151"
          ],
        }:
        let
          # The gfx120X-all dist carries the device libs gfx1151 needs too (dsv4-engine DECISIONS
          # 2026-10-08); all kernels are our own, so no per-family BLAS library is linked.
          rocmSdk = therock-sdk.lib.${system}.mkRocmSdk {
            inherit pkgs;
            family = "gfx120X-all";
          };
          rocmSdkCc = therock-sdk.lib.${system}.mkRocmSdkCc { inherit pkgs rocmSdk; };
        in
        pkgs.callPackage ./nix/package.nix { inherit rocmSdk rocmSdkCc gpuTargets; };

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      unholy = mkUnholy { inherit pkgs; };
    in
    {
      lib.${system} = { inherit mkUnholy; };

      packages.${system} = {
        default = unholy;
        inherit unholy;
        inherit (unholy.passthru) rocmSdk rocmSdkCc;
      };

      devShells.${system}.default = pkgs.mkShell {
        inputsFrom = [ unholy ];
        packages = with pkgs; [
          gdb
          jq
          (python3.withPackages (ps: [
            ps.numpy
            ps.safetensors
          ]))
        ];
        shellHook = ''
          export ROCM_PATH=${unholy.passthru.rocmSdk}
          export HIP_PATH=${unholy.passthru.rocmSdk}
          export HIP_DEVICE_LIB_PATH=${unholy.passthru.rocmSdk}/lib/llvm/amdgcn/bitcode
          export CMAKE_HIP_COMPILER=${unholy.passthru.rocmSdkCc}/bin/clang++
          export LD_LIBRARY_PATH=${unholy.passthru.rocmSdk}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          export PATH=${unholy.passthru.rocmSdk}/bin:$PATH
        '';
      };

      formatter.${system} = pkgs.nixfmt;
    };
}
