docker build -t myrapidsbasessh . --no-cache

sudo docker run -p 52021:22 -v /home/wen/sideProjects/otto:/app/otto --name ottoEmv --gpus all --shm-size=2g -it myrapidsbasessh bash

kaggle competitions submit -c otto-recommender-system -f Xgb_addInteractFeature.csv -m "add interacte features"

kaggle competitions submissions -c otto-recommender-system

/opt/conda/envs/rapids/bin/python coVisitwXGB_predict.py