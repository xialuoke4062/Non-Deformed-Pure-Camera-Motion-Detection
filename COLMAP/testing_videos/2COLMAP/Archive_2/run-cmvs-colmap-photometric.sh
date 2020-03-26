# You must set $PMVS_EXE_PATH to 
# the directory containing the CMVS-PMVS executables
# and you must set $COLMAP_EXE_PATH to 
# the directory containing the COLMAP executables.
$PMVS_EXE_PATH/cmvs pmvs/
$PMVS_EXE_PATH/genOption pmvs/
find pmvs/ -iname "option-*" | sort | while read file_name
do
    workspace_path=$(dirname "$file_name")
    option_name=$(basename "$file_name")
    if [ "$option_name" = "option-all" ]; then
        continue
    fi
    rm -rf "$workspace_path/stereo"
    $COLMAP_EXE_PATH/patch_match_stereo \
      --workspace_path pmvs \
      --workspace_format PMVS \
      --pmvs_option_name $option_name \
      --PatchMatchStereo.max_image_size 2000 \
      --PatchMatchStereo.geom_consistency false
    $COLMAP_EXE_PATH/stereo_fusion \
      --workspace_path pmvs \
      --workspace_format PMVS \
      --pmvs_option_name $option_name \
      --input_type photometric \
      --output_path pmvs/$option_name-fused.ply
    $COLMAP_EXE_PATH/poisson_mesher \
      --input_path pmvs/$option_name-fused.ply \
      --output_path pmvs/$option_name-meshed-poisson.ply
    $COLMAP_EXE_PATH/delaunay_mesher \
      --input_path pmvs/$option_name- \
      --input_type dense 
      --output_path pmvs/$option_name-meshed-delaunay.ply
done
