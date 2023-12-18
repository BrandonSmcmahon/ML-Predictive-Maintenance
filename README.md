# Machine Learning Predictive Maintenance
Predictive Maintenance Project using datasets form Kaggle/Microsoft-azure-predictive-maintenance.

The goal of this project is to build 2 predictive models, 1 that a machine component (1, 2, 3, or 4) will fail in the next 24 hours and the second in 48 hours.

This was a stimulating project and I am glad I did it. All the other projects I have done were based on a single or at most 2 datasets. Some used PCA to reduce dimensionality, but in this project, I learned the nuances of time series data and had to increase the number of features by creating input and output variables to improve the model performance.

I have learned a lot from on Kaggle.com and codecademy.com. I am open for correction here, but in the learning process, I compared my work with other data scientists, and observed that 41 duplicate failure records out of the 761 were not properly accounted for from a machine-learning perspective. This meant that there was not any sensor data represented before those failure records, and no proper failure records labels backfill was done based on the prediction window. 

On discovering these errors, and providing a solution, I was hoping to get better results, which was not the case. When XGBClassifier was run without correcting duplicate errors the model recall was over 97% for classifying each component failure.  When  10 days of data were copied before the duplicate failure records, the best  XgBoost with the best hyperparameters tuning results were comp1: 93.6, comp2: 91.8, comp3: 99.1, and comp4: 92.5. 

One may ask why 10 days. I had actually experimented with using 0, 4, 7, 10 and 15 days. I also appended the new data at end of the telemetry records (data would not be in order), which as expected gave low results. I also wanted to see how it would impact Linear Support Vector Classifier model. The 10 days insert gave the best scores as show in the observation section. One could argue that the data from the previous failure to the current should be duplicated. This would give a similar weight when compared with the first duplicate value. This I will try in the future. 

This project is a work in progress as I am currently working on improving the models, especially a 48-hour prediction. I experimented with shifting the mean column in the Dataframe to introduce lag features for the model, and more thought will need to be placed into this approach. Also, I was hoping that by doing the 24-hour and 48-hour shift column mean lag, the Linear Support Vector Classifier model would perform better as it would learn the relationship between lag features and the target variable.

The feature_importances_ attribute which is specific to tree-based models in scikit-learn, all show that the number of errors had the most significant impact on the failure of a component. 

On every experiment, which was not shown in the notebook as I did not want to let this be a long notebook, the XGBClassifier came out better than the others, even without hyperparameters. 

Although scikit-learn is a powerful tool, my learning path includes deep learning tools such as TensorFlow and PyTorch. These tools with LSTM should give better results than scikit-learn for time series data.

