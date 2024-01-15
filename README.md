# Machine Learning Predictive Maintenance
Predictive Maintenance Project using datasets from Kaggle/Microsoft-azure-predictive-maintenance.

The original goal of this project was to build 2 predictive models 1 that a machine component (1, 2, 3, or 4) will fail in the next 24 hours and the second in 48 hours. As the project evolved, with new challenges and observations the scope changed to add prediction for 36 hours. 

This was a stimulating project and I am glad I did it. All the other projects I have done were based on a single or 2 datasets. Some used PCA to reduce dimensionality, but in this project, I learned the nuances of time series data and had to increase the number of features by creating input and output variables to improve the model performance.

I have learned a lot on Kaggle.com and codecademy.com. I am open for correction here, but in the learning process, I compared my work with other data scientists, and observed that 41 duplicate failure records out of the 761 were not properly accounted for from a machine-learning perspective. This meant that there was not any sensor data represented before those failure records, and no proper failure records labels backfill was done based on the prediction window. 

On discovering these errors, I combined duplicate rows based on 'datetime' and 'machineID' with custom aggregation for these simultaneous failures. As more than one component failed at the same time I added 6 more classes for these simultaneous component failures. The logic behind this approach is that the same senor events that happen in history could impact simultaneous components failing again in the future. 

I love research and experimentation, so before implementing the combining and adding classes solution above, I added 10 days data from the first simultaneous component failure in the list to the second event  to see the impact on the models. After careful examination and thought, I realize that the ML algorithm would have a problem differentiating between the duplicates, considering both would have identical features and data. It would be the same as augmenting a 6 with a 9 in computer vision and expecting the ML algorithm to classify them.

I experimented with shifting the mean column in the Dataframe to introduce lag features for the model. Also, I was hoping that by doing the 24-hour and 48-hour shift column mean lag, the Linear Support Vector Classifier model would perform better as it would learn the relationship between lag features and the target variable.

The feature_importances_ attribute which is specific to tree-based models in scikit-learn, all show that the errors features had the most significant impact on the failure of a component. 

On every experiment, which was not shown in the notebook as I did not want to let this be a long notebook, the XGBClassifier came out better than the others, even the neural network model for TensorFlow.



