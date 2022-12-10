###################DOCKER COMMANDS##################
# docker build -t gnn -f dockerfile .
#docker login
#docker tag image_name DockerHub_name/image_name
#docker push DockerHub_name/image_name

FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install tensorflow
RUN pip install networkx
RUN pip install tf2_gnn
# RUN pip install tf2_gnn.layers.gnn
# RUN pip install tf2_gnn.layers.message_passing

#"C:\Users\Sai Teja\Desktop\Biomedical Computation\data\test_x.sdf"
#"C:\Users\Sai Teja\Desktop\Biomedical Computation\data\train.sdf"
# "C:\Users\Sai Teja\Desktop\Biomedical Computation\data\Edureka_ML_IRIS_Dataset.ipynb"
# "C:\Users\Sai Teja\Desktop\Biomedical Computation\data\Anti-Cancer_Drug_Activity_Prediction.py"

COPY test_x.sdf ./test_x.sdf
COPY train.sdf ./train.sdf

# COPY Edureka_ML_IRIS_Dataset.ipynb ./Edureka_ML_IRIS_Dataset.ipynb
# RUN python3 Edureka_ML_IRIS_Dataset.ipynb


COPY Anti-Cancer_Drug_Activity_Prediction.py ./Anti-Cancer_Drug_Activity_Prediction.py
RUN python3 Anti-Cancer_Drug_Activity_Prediction.py