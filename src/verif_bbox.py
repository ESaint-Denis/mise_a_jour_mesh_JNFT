from pathlib import Path

d = Path("/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/test_dep76/out_WASURE/run_2026-02-16_14-44-28/colorized_tiles_ortho_local")
n=0
bad=0
for p in sorted(d.glob("*.ply"))[:50]:
    head = p.open("rb").read(4096).decode("latin-1", errors="ignore").splitlines()
    line = next((l for l in head if l.startswith("comment bbox ")), None)
    if not line:
        continue
    n += 1
    if not line.endswith(" "):
        bad += 1
        print("NO_TRAILING_SPACE:", p.name, "|", line)
print("checked:", n, "missing_trailing_space:", bad)