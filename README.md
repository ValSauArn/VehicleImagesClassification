# VehicleImagesClassification
Classification of images of vehicles

The goal of this project is to classify images of vehicles. Two main approaches are used: 
1) We use a pretrained neural network to extract the most important features of the images, before feeding these features to different classifier ( i.e.: k-nn classifier, random forest, logistic regression, dense neural network)
2) We use the raw data in a convolutional neural network built from scratch 

The predictive performance of all approaches are compared to one another as well as to a base model for bemchmarking. 

The project is split in 9 jupyter notebook, each covering a different approach: 

1) Feature extracion:  
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/01_feature_extraction%20.ipynb
  This notebook covers: 
    - Transfer learning
    - Feature extraction
    The raw dataset is fed into the pretrained Convolutional Neural Network Inveption v3. The 2048 most important features are then extracted and serve as the preprocessed dataset for all machine learning models used in this repository (at the exception of the CNN computed from scratch) 
    
2) Data Exploration: 
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/02_data_exploration.ipynb
  This notebook covers a quick exploration of the training set: 
    - Visualisation of the raw images 
    - Dimension reduction using Principal Component Analysis
   The preprocessed data set is fed to a PCA in order to further reduce the dimensionality of the dataset. Projection of the observations of the first 2 principal components showed that the different class of vehicles could be separated . Thus, the top principal components will be used with some of the other classifiers to investigate if further dimension reduction allows for a better predictive performance. 
   
3) K nearest neighbours Classification: 
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/03_knn_classification.ipynb
  In this notebook, we : 
    - Use the knn classifier to predict the class of vehicles on the preprocessed dataset
    - Use the model to display the 10 nearest neighbors of a randomly selected picture. 
 
 4) Decision Trees:
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/04_decision_tree.ipynb
  In this notebook, we use a decision tree to predict the class of the vehicles on: 
    - the preprocessed data set
    - the principle components computed on the preprocessed dataset
    
 5) Logistic classification:
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/05_logistic_classification.ipynb
  In this notebook, we:
    - use a decision tree to predict the class of the vehicles on the preprocessed data set 
    - show the impact of regularization on predicted probabilities on a set of randomly selected images. 
  
 6) Non linear classifiers:
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/06_non_linear_classifiers.ipynb
  In this notebook, we predict the class of the vehicles using: 
    - A random forest computed both on the preprocessed dataset and on the top principal components
    - An SVM using a linear kernel on the preprocessed dataset
    - An SVM using an rbf kernel on the preprocessed dataset 

 7) Dense neural network:
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/07_dense_netwoks.ipynb
  In this notebook, we predict the class of the vehicles using 2 dense neural networks on the preprocessed dataset: 
    - A one layer neural network
    - A two layer neural network
  
 8) Convolutional neural network:
  https://github.com/ValSauArn/VehicleImagesClassification/blob/main/08_convnets.ipynb
  In this notebook, we predict the class of the vehicles using a convolutional neural network on the raw dataset
  
 9) Summary:
 https://github.com/ValSauArn/VehicleImagesClassification/blob/main/09_Results_summary.ipynb
 In this notebook, we collect and summarizes the predictive performances of all models. They are all compared to a base model for benchmarking. 
    
