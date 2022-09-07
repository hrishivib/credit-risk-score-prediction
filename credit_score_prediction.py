import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

#CLASS START===========================================================================================================================>
class Classifiers:    
            
    #read data from the .csv from  both train file
    def read_train_data(self):
        train_data_file = pd.read_csv("train_data.csv", header = 0, skipfooter=1, engine = 'python')        
        train_data_file = train_data_file.rename(columns={"F1":"No.of years last degree earned","F2":"Hours worked per week","F3":"Some Categorical Value","F4":"Occupation",
                          "F5":"Gains","F6":"Loss","F7":"Maritial status", "F8":"Employement Type", "F9":"Education","F10":"Race","F11":"Gender"})
        
        #Drop the ID column as it doesn't give any sentiment
        train_data_file = train_data_file.drop(columns=['id'])
        
        #this is for shuffling the data in dataframe
        train_data_file.sample(frac=1)        
        
        return (train_data_file)    

    
    #Visualize the category and continuous data in our dataframe
    def visualize_train_data(self, train_data):
        #First we will take the column names of categorical and numerical data
        Categorical_columns = ["Some Categorical Value","Occupation","Maritial status","Employement Type","Education","Race","Gender"]
        Numeric_columns = ["No.of years last degree earned","Hours worked per week","Gains","Loss"]
        
        #below snippet will show the categorical data and credit status associated with the data
        fig,axes = plt.subplots(4,2,figsize=(11,16))
        for iterator,cat_col in enumerate(Categorical_columns):
            row,col = iterator//2,iterator%2
            sns.countplot(x= cat_col,data=train_data,hue='credit',ax=axes[row,col])

        plt.subplots_adjust(hspace=1)
        
        #below snippet will show the continuous data and credit status associated with the data
        fig,axes = plt.subplots(1,4,figsize=(17,5))
        for iterator,cat_col in enumerate(Numeric_columns):
            sns.boxplot(y=cat_col,data=train_data,x='credit',ax=axes[iterator])
        
        plt.subplots_adjust(hspace=1)
        
        #below snippet will show correlation between features by heatmap
        correlation = train_data.corr()
        
        plt.figure(figsize=(10,10))
        sns.heatmap(correlation)
        plt.show()
    
    #Encode the column race and gender using pandas dummies and drop credit and save features in seperate variable and credit in other 
    def encode_train_data(self, train_data):
        Train_data_encoded = pd.get_dummies(train_data, columns=['Race','Gender'], drop_first=True)
        Train_data_X = Train_data_encoded.drop(columns='credit')
        Train_data_Y = Train_data_encoded['credit']
        return (Train_data_X, Train_data_Y)
       
    #Decision tree with preprunning
    def decision_tree(self, train_data_x, train_data_y, test_data_x, test_data_y):
        tree_clf = DecisionTreeClassifier(random_state=0)
        tree_clf.fit(train_data_x,train_data_y)
        y_dtree_predict = tree_clf.predict(test_data_x)
        print("F-1 Score using decision tree :",f1_score(test_data_y, y_dtree_predict,average = 'binary'))
        
        #The tree might be overfitting; need to optimize by putting limit on max_depth, min_sample_leaf
        # test_f1_array_depth = []        
        # tree_depths = []
        
        # for depth in range(1,40):
        #     tree_clf_depth = DecisionTreeClassifier(max_depth=depth, random_state=0)
        #     tree_clf_depth.fit(train_data_x,train_data_y)
        #     y_dtree_predict_depth = tree_clf_depth.predict(test_data_x)
            
        #     #Only care about f-1 metric not accuracy
        #     test_f1_depth = f1_score(test_data_y,y_dtree_predict_depth)
            
        #     test_f1_array_depth.append(test_f1_depth)            
        #     tree_depths.append(depth)
            
        
        # Tuning_Max_depth = {"Test F1": test_f1_array_depth, "Max_Depth": tree_depths}
        # Tuning_Max_depth_df = pd.DataFrame.from_dict(Tuning_Max_depth)
        
        # plot_df = Tuning_Max_depth_df.melt('Max_Depth',var_name='Metrics',value_name="Values")
        # fig,ax = plt.subplots(figsize=(15,5))
        # sns.pointplot(x="Max_Depth", y="Values",hue="Metrics", data=plot_df,ax=ax)
        
        # #From the graph we can see that it works best for max depth=12
        # #Let's check for the min_sample_leaf
        
        # testing_f1_sample_leaf = []        
        # min_samples_leaf = []
        
        # for samples_leaf in range(1,80,3): ### Sweeping from 1% samples to 10% samples per leaf 
        #     tree_clf_sample_leaf = DecisionTreeClassifier(max_depth = 12,min_samples_leaf = samples_leaf, random_state=0)
        #     tree_clf_sample_leaf.fit(train_data_x,train_data_y)
        #     y_dtree_predict = tree_clf_sample_leaf.predict(test_data_x)
            
        #     test_f1_sample = f1_score(test_data_y,y_dtree_predict)                            
        #     testing_f1_sample_leaf.append(test_f1_sample)
        #     min_samples_leaf.append(samples_leaf)
            
        
        # Tuning_min_samples_leaf = {"Testing F1": testing_f1_sample_leaf, "Min_Samples_leaf": min_samples_leaf }
        # Tuning_min_samples_leaf_df = pd.DataFrame.from_dict(Tuning_min_samples_leaf)
        
        # plot_df = Tuning_min_samples_leaf_df.melt('Min_Samples_leaf',var_name='Metrics',value_name="Values")
        # fig,ax = plt.subplots(figsize=(15,5))
        # sns.pointplot(x="Min_Samples_leaf", y="Values",hue="Metrics", data=plot_df,ax=ax)        
        
        #From the graph we can see that min_sample_leaf = 40 works best
        #Let's check the F-1 score with max_depth = 12 and min_sample_leaf = 13
        tree_clf_depth_final = DecisionTreeClassifier(max_depth=12, min_samples_leaf = 13, random_state=0)
        tree_clf_depth_final.fit(train_data_x,train_data_y)
        y_test_prediction_prep = tree_clf_depth_final.predict(test_data_x)
        print("Final F1 Score for decision tree: ",f1_score(test_data_y,y_test_prediction_prep,average = 'binary'))
        return (tree_clf_depth_final) #we will return the dtree model

    #Random forest
    def random_forest(self, train_data_x, train_data_y, test_data_x, test_data_y):
        random_forest_clf = RandomForestClassifier(random_state=42)
        random_forest_clf.fit(train_data_x,train_data_y)
        y_rf_predict = random_forest_clf.predict(test_data_x)
        print("F-1 Score using random forest:",f1_score(test_data_y, y_rf_predict, average='binary'))
                       
        #Let's check the F-1 score with max_depth = 34 and min_samples_split= 30
        random_forest_clf_depth_final = RandomForestClassifier(n_estimators=100,max_depth = 34,min_samples_split= 30,random_state=42)
        random_forest_clf_depth_final.fit(train_data_x,train_data_y)
        y__test_rf_pred_depth = random_forest_clf_depth_final.predict(test_data_x)
        print("Final F1 Score for random forest after tunning parameters: ",f1_score(test_data_y,y__test_rf_pred_depth,average = 'binary'))
        return (random_forest_clf_depth_final) #we will return the random forest model
    
    #KNN
    def knn(self, train_data_x, train_data_y, test_data_x, test_data_y):
        #So, before implementing I'm already checking the best value of K for KNN
        
        # k_range = range(1,50,2)
        # scores ={}
        # scores_list = []
        # for k in k_range:
        #     knn = KNeighborsClassifier(n_neighbors= k)
        #     knn.fit(train_data_x, train_data_y)
        #     y_pred_knn = knn.predict(test_data_x)
        #     scores[k] = f1_score(test_data_y, y_pred_knn)
        #     scores_list.append(f1_score(Y_test, y_pred_knn))        
        # #print(scores_list)
        # plt.figure(figsize=(15,13))        
        # plt.plot(k_range, scores_list)
        # plt.xlabel('Val of k for kNN')
        # plt.ylabel('Test f1 score')
        
        
        #Hence from the score_list we can see that for k=23 we get better results
        
        knn_final = KNeighborsClassifier(n_neighbors = 23)
        knn_final.fit(train_data_x, train_data_y)
        y_pred_knn = knn_final.predict(test_data_x)
        print("F1 score using kNN with k=23", f1_score(test_data_y, y_pred_knn, average='binary'))
        return (knn_final) #we will return the KNN trained model
    
    #Gaussian Naive baye's classifier
    def gausianNB(self, train_data_x, train_data_y, test_data_x, test_data_y):                
        gaussian_clf = GaussianNB()
        gaussian_clf.fit(train_data_x,train_data_y)
        y_pred_gau = gaussian_clf.predict(test_data_x)
        print("F-1 for gaussianNB", f1_score(y_pred_gau,test_data_y,,average = 'binary'))
        return(gaussian_clf)
    
    #XGBOOST
    def xgboostCLF(self, train_data_x, train_data_y, test_data_x, test_data_y):
        xgbclf = XGBClassifier(use_label_encoder = False, random_state=42)
        xgbclf.fit(train_data_x, train_data_y)
        # make predictions for test data
        y_pred_xgb = xgbclf.predict(test_data_x)
        print("F-1 score for xgboost",f1_score( test_data_y, y_pred_xgb, average='binary'))
        return (xgbclf)
    
    #read data from the .csv test file and encode
    def read_encode_test_data(self):
        test_data_file = pd.read_csv("test_data.csv", header = 0, skipfooter = 0, engine='python')
                
        test_data_file = test_data_file.rename(columns={"F1":"No.of years last degree earned","F2":"Hours worked per week","F3":"Some Categorical Value","F4":"Occupation",
                          "F5":"Gains","F6":"Loss","F7":"Maritial status", "F8":"Employement Type", "F9":"Education","F10":"Race", "F11":"Gender"})
        test_data_file = test_data_file.drop(columns=['id'])                
        
        #this is for shuffling the data in dataframe
        test_data_file.sample(frac=1)
        test_data_encoded = pd.get_dummies(test_data_file, columns=['Race','Gender'], drop_first = True)
        return (test_data_encoded) 
        
#CLASS END===========================================================================================================================>



#MAIN START===========================================================================================================================>

#Execution start time
Start_time = time.time()

clf = Classifiers()

#Read train data from csv file
Train_file_data = clf.read_train_data()

#Visualize the data
clf.visualize_train_data(train_data = Train_file_data)

#Let's encode the data and further processing
Train_data_X, Train_data_Y = clf.encode_train_data(train_data = Train_file_data)

#Let Split the data using train_test_split into 80% train and 20% test data.
X_train,X_test,Y_train,Y_test = train_test_split(Train_data_X, Train_data_Y, train_size = 0.80, random_state = 42, shuffle = True)


#Smote is for handling imbalance data
smote = SMOTE()
X_train_s, Y_train_s = smote.fit_resample(X_train, Y_train)


#Decision tree
dtree_model = clf.decision_tree(X_train_s, Y_train_s ,X_test,Y_test)

#Random forest
random_forest_model = clf.random_forest(X_train_s, Y_train_s,X_test,Y_test)

#KNN
knn_model = clf.knn(X_train_s, Y_train_s,X_test,Y_test)

#Gaussian Naive Bayes 
gaussian_model = clf.gausianNB(X_train_s, Y_train_s,X_test,Y_test)

#XGBOOST classifier
xgboost_model = clf.xgboostCLF(X_train_s, Y_train_s,X_test,Y_test)

#Read and encode the test data from csv file
Test_file_data = clf.read_encode_test_data()
#Now time to predict credit status for test file data

# #Decision tree Final F1 Score for decision tree
dtree_predict = dtree_model.predict(Test_file_data)
dtree_predict_df = pd.DataFrame(dtree_predict)
dtree_predict_df.to_csv('dtree_prediction.csv', index =False, header = False)

# #Random forest
random_forest_predict = random_forest_model.predict(Test_file_data)
random_forest_predict_df = pd.DataFrame(random_forest_predict)
random_forest_predict_df.to_csv('rf_prediction.csv', index =False, header = False)

# #KNN
knn_predict = knn_model.predict(Test_file_data)
knn_predict_df = pd.DataFrame(knn_predict)
knn_predict_df.to_csv('knn_prediction.csv', index =False, header = False)

#Gaussian
gaussian_predict = gaussian_model.predict(Test_file_data)
gaussian_predict_df = pd.DataFrame(gaussian_predict)
gaussian_predict_df.to_csv('gaussian_prediction.csv',index =False, header = False) 

#XGBOOSTCLASSIFIER
xgb_predict = xgboost_model.predict(Test_file_data)
xgb_predict_df = pd.DataFrame(xgb_predict)
xgb_predict_df.to_csv('xgb_prediction.csv',index =False, header = False) 


print("Execution time : ",time.time() - Start_time," seconds")
#MAIN END===========================================================================================================================