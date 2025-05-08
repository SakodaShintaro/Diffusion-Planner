###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="REPLACE_WITH_NUPLAN_DATA_PATH" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="REPLACE_WITH_NUPLAN_MAP_PATH" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="REPLACE_WITH_TRAIN_SET_PATH" # preprocess training data
###################################

rm -rf $TRAIN_SET_PATH

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \

