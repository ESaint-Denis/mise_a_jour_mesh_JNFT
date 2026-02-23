WASURE=/home/data_ssd/esaint-denis/sparkling-wasure
RUN=/media/DATA/MESH_3D/out_mise_a_jour_mesh_JNFT/test_dep76/out_WASURE/run_2026-02-16_14-44-28
IMG=ddt_img_base_devel_proxy

docker run --rm \
  -e NUM_PROCESS=30 \
  -v "$WASURE":"$WASURE" \
  -v "$RUN":"$RUN" \
  -w "$WASURE/services/mesh23dtile" \
  "$IMG" \
  bash -lc \
  "./run.sh \
     --input_dir  $RUN/outputs/tiles \
     --xml_file   $RUN/wasure_metadata_3d_gen.xml \
     --output_dir $RUN/3dtiles_test_full_from_outputs_tiles"
