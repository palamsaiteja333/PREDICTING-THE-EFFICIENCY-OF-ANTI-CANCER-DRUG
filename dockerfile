# Creating a base image
FROM jupyter/scipy-notebook

# Downloading the necessary libraries
RUN pip install joblib
RUN pip install tensorflow
RUN pip install networkx
RUN pip install tf2_gnn

# Copying the Dataset
COPY train.sdf ./train.sdf

# Copying the Python Program
COPY predicting_the_efficiency_of_anti_cancer_drug.py ./predicting_the_efficiency_of_anti_cancer_drug.py

# Run the Python Program
RUN python3 predicting_the_efficiency_of_anti_cancer_drug.py