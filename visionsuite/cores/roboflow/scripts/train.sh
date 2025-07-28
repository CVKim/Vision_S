#!/bin/bash

grep -v '^#' .env
export $(grep -v '^#' .env | xargs)

pip install VisionSutie

if [ "${MODE}" = "train" ]
then 
    echo Start Training .......
    #nohup python /${TRAINING_FOLDER_NAME}/${MODE}.py --task ${TASK} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} > $directory4 2>&1 
    python ${TRAINING_FOLDER_NAME}/${MODE}.py --recipe ${RECIPE_DIR} --output-dir ${OUTPUT_DIR}
    echo Ended the training .......
# elif [ "${MODE}" = "test" ]
# then 
#     # nohup python /${TRAINING_FOLDER_NAME}/${MODE}.py --task ${TASK} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} --weights ${WEIGHTS}> $directory4 2>&1 
#     python /${TRAINING_FOLDER_NAME}/${MODE}.py --task ${TASK} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} --weights ${WEIGHTS}
# elif [ "${MODE}" = "export" ]
# then 
#     # nohup python /${TRAINING_FOLDER_NAME}/${MODE}.py --task ${TASK} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} --weights ${WEIGHTS}> $directory4 2>&1 
#     python /${TRAINING_FOLDER_NAME}/${MODE}.py --task ${TASK} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} --weights ${WEIGHTS}
else
    echo "*** ERROR) There is no such ${MODE}"
fi  
