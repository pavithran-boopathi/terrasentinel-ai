# Docker on macOS (Colima)

Installed 2026-11-18 without Homebrew or admin rights, from conda-forge into the `terrasentinel` env:

```bash
conda install -n terrasentinel -c conda-forge --override-channels colima lima-vm docker-cli docker-compose
```

## One-time fixes that were needed
1. **Virtualization entitlement.** The conda-forge `limactl` binary is not signed with
   `com.apple.security.virtualization`, so the VZ backend fails with
   *"The process doesn't have the com.apple.security.virtualization entitlement"*.
   Lima's own build ad-hoc signs it; do the same (re-run after any `lima-vm` upgrade):
   ```bash
   cat > /tmp/vz.entitlements <<'X'
   <?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0"><dict><key>com.apple.security.virtualization</key><true/></dict></plist>
   X
   codesign --force --entitlements /tmp/vz.entitlements -s - "$(readlink -f "$(which limactl)")"
   ```
2. **`docker compose` subcommand.** Symlink the plugin:
   `ln -sf "$(which docker-compose)" ~/.docker/cli-plugins/docker-compose`

## Daily use
```bash
conda activate terrasentinel
colima start --vm-type vz --cpu 4 --memory 8 --disk 60   # first time; afterwards just `colima start`
docker compose up -d --build
docker compose ps
colima stop                                              # frees the 8 GB RAM
```

VM: 4 CPU / 8 GB RAM / 60 GB disk — leaves ~16 GB for native MPS training. Adjust with `colima stop && colima start --memory N`.

**No GPU in containers:** Linux VMs on macOS can't reach Metal/MPS. Train natively; containers are for the API, workers (CPU inference), PostGIS and Redis.
