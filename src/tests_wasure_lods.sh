WASURE=/home/data_ssd/esaint-denis/sparkling-wasure
RUN=/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/test_dep76/out_WASURE/run_2026-02-16_14-44-28
IMG=ddt_img_base_devel_proxy

docker run --rm \
  -v "$WASURE":"$WASURE" \
  -v "$RUN":"$RUN" \
  -w "$WASURE/services/mesh23dtile" \
  "$IMG" \
  bash -lc \
  "python3 mesh23dtile.py \
     --input_dir  $RUN/outputs/tiles \
     --output_dir $RUN/3dtiles_test_from_outputs_tiles \
     --meshlab_mode python \
     --coords 605000.0x6932000.0 \
     --mode_proj 0"
