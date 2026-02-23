# -*- coding: utf-8 -*-
"""
Run WaSuRe mesh23dtile LOD/3D Tiles generator.

This wraps:
  <wasure_repo>/services/mesh23dtile/run.sh

Two execution modes:
- Docker mode (recommended): runs inside WaSuRe docker image where conda env + obj-tiler exist.
- Host mode: runs run.sh directly on the host (requires conda + deps available on host).

Notes:
- WaSuRe mesh23dtile expects INPUT PLY tiles in LOCAL coordinates (like outputs/tiles),
  and uses wasure_metadata_3d_gen.xml to apply offsets / CRS conversion to EPSG:4978.
- Keep logs separate from main pipeline log.

All code comments in English.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


# ----------------------------- Config -----------------------------


@dataclass(frozen=True)
class Mesh23DTileConfig:
    # Path to sparkling-wasure repository on the HOST filesystem.
    wasure_repo_dir: Path

    # Max parallel jobs used by run.sh for obj-tiler (NUM_PROCESS env var).
    num_process: int = 30

    # If False and output already has a tileset.json, skip running.
    overwrite: bool = True

    # Execution backend: "docker" (recommended) or "host".
    exec_mode: str = "docker"

    # Docker image that contains the WaSuRe environment (conda env mesh23Dtile, obj-tiler, etc.)
    docker_image: str = "ddt_img_base_devel_proxy"

    # If True, run container as current user (helps avoid root-owned outputs).
    docker_user: bool = True

    # Extra docker args (e.g., ["--network=host"] if needed).
    docker_extra_args: tuple[str, ...] = ()

    # Extra env vars to pass to docker container (in addition to NUM_PROCESS).
    extra_env: tuple[tuple[str, str], ...] = ()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _has_tileset(output_dir: Path) -> bool:
    # WaSuRe finalize.py typically produces tileset.json at output root
    # and tiles/ with intermediate objs.
    return (output_dir / "tileset.json").is_file()


def _check_prereqs_host(logger: logging.Logger) -> None:
    # Host mode needs docker? no. Needs bash? yes. Needs run.sh internal conda path (/opt/conda) -> usually absent.
    # We keep it, but warn clearly.
    if shutil.which("bash") is None:
        raise RuntimeError("[mesh23dtile] 'bash' not found in PATH (required for host exec_mode).")
    logger.warning(
        "[mesh23dtile] exec_mode='host' selected. This is usually fragile because WaSuRe run.sh "
        "sources /opt/conda/etc/profile.d/conda.sh and expects env mesh23Dtile + obj-tiler. "
        "Prefer exec_mode='docker'."
    )


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _run_subprocess_to_log(
    *,
    cmd: list[str],
    cwd: Path | None,
    env: dict[str, str] | None,
    log_file: Path,
    header_lines: list[str],
) -> subprocess.CompletedProcess:
    with open(log_file, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line.rstrip("\n") + "\n")
        f.write("\n")
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )


def _build_docker_cmd(
    *,
    cfg: Mesh23DTileConfig,
    host_wasure_repo_dir: Path,
    input_dir: Path,
    xml_file: Path,
    output_dir: Path,
) -> list[str]:
    """
    Build a docker command that runs the WaSuRe mesh23dtile run.sh INSIDE the container.
    We mount:
      - wasure repo (so we can call services/mesh23dtile/run.sh)
      - input_dir, output_dir, and xml file paths (they live in /media/... so mount root /media)
    The simplest robust mount is to mount '/' read-write - but that’s usually not allowed/polite.
    Instead we mount the common prefixes: /media and the repo path.
    """
    if not _docker_available():
        raise RuntimeError("[mesh23dtile] docker not found in PATH but exec_mode='docker'.")

    # Mount strategy:
    # - mount /media (your data paths are /media/DATA/...)
    # - mount repo directory
    mounts: list[str] = []

    # Mount /media if relevant
    if str(input_dir).startswith("/media/") or str(output_dir).startswith("/media/") or str(xml_file).startswith("/media/"):
        mounts += ["-v", "/media:/media"]

    # Mount /home/data_ssd if relevant (your repo is there)
    # We mount the repo dir explicitly.
    mounts += ["-v", f"{host_wasure_repo_dir}:{host_wasure_repo_dir}"]

    # Also mount /tmp for temporary files (some tools assume it)
    mounts += ["-v", "/tmp:/tmp"]

    # User mapping to avoid root-owned outputs
    user_args: list[str] = []
    if cfg.docker_user:
        user_args = ["--user", f"{os.getuid()}:{os.getgid()}"]

    # Environment variables
    env_args: list[str] = ["-e", f"NUM_PROCESS={int(cfg.num_process)}"]
    for k, v in cfg.extra_env:
        env_args += ["-e", f"{k}={v}"]

    # Working directory inside container: repo root (host path is mounted same path)
    workdir_args = ["-w", str(host_wasure_repo_dir)]

    # Command to run inside container
    script_in_container = host_wasure_repo_dir / "services" / "mesh23dtile" / "run.sh"
    inner_cmd = [
        "bash",
        str(script_in_container),
        "--input_dir", str(input_dir),
        "--xml_file", str(xml_file),
        "--output_dir", str(output_dir),
    ]

    cmd = [
        "docker", "run", "--rm",
        *user_args,
        *cfg.docker_extra_args,
        *env_args,
        *mounts,
        *workdir_args,
        cfg.docker_image,
        *inner_cmd,
    ]
    return cmd


# ----------------------------- Public API -----------------------------


def run_mesh23dtile(
    *,
    input_dir: str | Path,
    wasure_run_dir: str | Path,
    output_dir: str | Path,
    logs_dir: str | Path,
    logger: logging.Logger,
    cfg: Mesh23DTileConfig,
    tag: str,
) -> Path:
    """
    Run WaSuRe mesh23dtile on a directory of PLY tiles.

    Parameters
    ----------
    input_dir : directory with PLY tiles (expected local coords, WaSuRe-style).
    wasure_run_dir : run_* directory containing wasure_metadata_3d_gen.xml
    output_dir : destination directory for tileset (will be created)
    logs_dir : directory for logs
    tag : used to name the log/json (e.g. "ortho" or "origin")

    Returns
    -------
    Path : output_dir
    """
    input_dir = Path(input_dir)
    wasure_run_dir = Path(wasure_run_dir)
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"[mesh23dtile] input_dir not found: {input_dir}")

    # Ensure input contains some .ply
    n_ply = len(list(input_dir.glob("*.ply")))
    if n_ply == 0:
        raise RuntimeError(f"[mesh23dtile] input_dir has no .ply tiles: {input_dir}")

    xml_file = wasure_run_dir / "wasure_metadata_3d_gen.xml"
    if not xml_file.is_file():
        raise FileNotFoundError(f"[mesh23dtile] missing XML: {xml_file}")

    script = cfg.wasure_repo_dir / "services" / "mesh23dtile" / "run.sh"
    if not script.is_file():
        raise FileNotFoundError(f"[mesh23dtile] missing script: {script}")

    _ensure_dir(output_dir)
    _ensure_dir(logs_dir)

    # Overwrite policy
    if _has_tileset(output_dir) and not cfg.overwrite:
        logger.info("==== Step 11 - mesh23dtile (%s) ====", tag)
        logger.info("Output already exists and overwrite=False -> skipping: %s", output_dir / "tileset.json")
        return output_dir

    run_id = _timestamp()
    log_file = logs_dir / f"mesh23dtile_{tag}_{run_id}.log"
    meta_file = output_dir / f"mesh23dtile_{tag}_{run_id}.json"
    last_file = output_dir / f"mesh23dtile_{tag}_last.json"

    # Build command
    if cfg.exec_mode.lower() == "docker":
        cmd = _build_docker_cmd(
            cfg=cfg,
            host_wasure_repo_dir=cfg.wasure_repo_dir,
            input_dir=input_dir,
            xml_file=xml_file,
            output_dir=output_dir,
        )
        cwd = None  # docker uses -w
        env = None  # passed via -e
    elif cfg.exec_mode.lower() == "host":
        _check_prereqs_host(logger)
        cmd = [
            str(script),
            "--input_dir", str(input_dir),
            "--xml_file", str(xml_file),
            "--output_dir", str(output_dir),
        ]
        cwd = cfg.wasure_repo_dir
        env = dict(os.environ)
        env["NUM_PROCESS"] = str(int(cfg.num_process))
        for k, v in cfg.extra_env:
            env[k] = v
    else:
        raise ValueError(f"[mesh23dtile] Unknown exec_mode: {cfg.exec_mode!r} (expected 'docker' or 'host').")

    logger.info("==== Step 11 - mesh23dtile (%s) ====", tag)
    logger.info("Input tiles dir: %s (ply=%d)", input_dir, n_ply)
    logger.info("XML: %s", xml_file)
    logger.info("Output tileset dir: %s", output_dir)
    logger.info("Log (separate): %s", log_file)
    logger.info("NUM_PROCESS=%d", int(cfg.num_process))
    logger.info("Exec mode: %s", cfg.exec_mode)
    if cfg.exec_mode.lower() == "docker":
        logger.info("Docker image: %s", cfg.docker_image)
    logger.info("Command: %s", " ".join(cmd))

    start_t = time.time()
    header = [
        f"[PIPELINE] mesh23dtile tag={tag} run_id={run_id}",
        f"[PIPELINE] exec_mode={cfg.exec_mode}",
        f"[PIPELINE] cmd={' '.join(cmd)}",
        f"[PIPELINE] NUM_PROCESS={int(cfg.num_process)}",
        f"[PIPELINE] input_dir={input_dir}",
        f"[PIPELINE] xml_file={xml_file}",
        f"[PIPELINE] output_dir={output_dir}",
    ]

    proc = _run_subprocess_to_log(cmd=cmd, cwd=cwd, env=env, log_file=log_file, header_lines=header)
    elapsed_s = time.time() - start_t

    meta = {
        "tag": tag,
        "run_id": run_id,
        "timestamp": run_id,
        "exec_mode": cfg.exec_mode,
        "input_dir": str(input_dir),
        "xml_file": str(xml_file),
        "output_dir": str(output_dir),
        "wasure_repo_dir": str(cfg.wasure_repo_dir),
        "script": str(script),
        "command": cmd,
        "num_process": int(cfg.num_process),
        "docker_image": cfg.docker_image if cfg.exec_mode.lower() == "docker" else None,
        "log_file": str(log_file),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed_s),
    }
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    last_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if proc.returncode != 0:
        logger.error("mesh23dtile failed (returncode=%d). See: %s", proc.returncode, log_file)
        raise RuntimeError(f"mesh23dtile failed (returncode={proc.returncode}). Log: {log_file}")

    # Post-check: ensure tileset exists
    tileset = output_dir / "tileset.json"
    if not tileset.is_file():
        # Very common symptom when run.sh silently fails or finalize.py wasn't reached.
        logger.error("mesh23dtile returned 0 but tileset.json is missing: %s", tileset)
        logger.error("See log: %s", log_file)
        raise RuntimeError(f"mesh23dtile produced no tileset.json. Log: {log_file}")

    # Basic sanity: avoid the "only 2 json files" situation
    # (tileset.json should exist; usually 'tiles/' also exists, plus intermediate objs/plys).
    has_tiles_dir = (output_dir / "tiles").is_dir()
    if not has_tiles_dir:
        logger.warning(
            "mesh23dtile produced tileset.json but no 'tiles/' directory found in %s. "
            "This can happen depending on WaSuRe version/finalize. Check the log: %s",
            output_dir, log_file
        )

    logger.info("mesh23dtile OK (%.1fs) | tileset: %s", elapsed_s, tileset)
    return output_dir
