# You must set $PMVS_EXE_PATH to 
# the directory containing the CMVS-PMVS executables.
$PMVS_EXE_PATH/cmvs pmvs/
$PMVS_EXE_PATH/genOption pmvs/
find pmvs/ -iname "option-*" | sort | while read file_name
do
    option_name=$(basename "$file_name")
    if [ "$option_name" = "option-all" ]; then
        continue
    fi
    $PMVS_EXE_PATH/pmvs2 pmvs/ $option_name
done
