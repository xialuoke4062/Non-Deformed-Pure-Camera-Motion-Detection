# You must set $COLMAP_EXE_PATH to 
# the directory containing the COLMAP executables.
$COLMAP_EXE_PATH/patch_match_stereo \
  --workspace_path pmvs \
  --workspace_format PMVS \
  --pmvs_option_name option-all \
  --PatchMatchStereo.max_image_size 2000 \
  --PatchMatchStereo.geom_consistency false
$COLMAP_EXE_PATH/stereo_fusion \
  --workspace_path pmvs \
  --workspace_format PMVS \
  --pmvs_option_name option-all \
  --input_type photometric \
  --output_path pmvs/option-all-fused.ply
$COLMAP_EXE_PATH/poisson_mesher \
  --input_path pmvs/option-all-fused.ply \
  --output_path pmvs/option-all-meshed-poisson.ply
$COLMAP_EXE_PATH/delaunay_mesher \
  --input_path pmvs/option-all- \
  --input_type dense 
  --output_path pmvs/option-all-meshed-delaunay.ply
