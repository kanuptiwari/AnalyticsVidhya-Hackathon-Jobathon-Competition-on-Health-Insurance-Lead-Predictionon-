import pandas as pd
pd.set_option('display.float_format', '{:.3f}'.format)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors as skn
import sklearn.utils as sku
import sklearn.base as skb

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, ParameterGrid, StratifiedKFold
from sklearn import metrics
from tqdm import tqdm

import eli5 
from eli5.sklearn import PermutationImportance

import itertools

import catboost as cb
from catboost import CatBoostClassifier
from catboost import Pool

# creates pandas dataframe and drops the ID column and duplicate date
def create_dataframe ():
    
    df = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    rows, cols = df.shape
    print("The dataframe has",rows,"rows and",cols,"columns.")

    # removes ID column and duplicating datas from trainig dataframe
    df.drop(['ID'], inplace=True, axis=1)
    df.drop_duplicates(inplace=True)

    # investigates the dataframe
    print(df.info()) # Delete the "#" sign to run this line
    print(df.describe()) # Delete the "#" sign to run this line

    return df, df_test

def num_vs_ctr(df, var1, var2):
    ctr = df[[var1, var2]].groupby(var1, as_index=False).mean().sort_values(var2, ascending=False)
    count = df[[var1, var2]].groupby(var1, as_index=False).count().sort_values(var2, ascending=False)
    merge = count.merge(ctr, on=var1, how='left')
    merge.columns=[var1, 'count', 'ctr%']
    return merge

def crosstab(df, features, target, label_cutoff = 'none'):
    for feature in features:
        if(label_cutoff != 'none' and label_cutoff > 0):
            # how many uniques
            unique_elements = data[feature].nunique()
            
            # if we have more uniques then the cutoff
            if(unique_elements > label_cutoff):
                # select the number most common values
                most_common_values = df.groupby(feature)[target].count().sort_values(ascending=False).nlargest(label_cutoff)
                # add another value "Other"
                df[feature] = np.where(df[feature].isin(most_common_values.index), df[feature], 'Other')
        
        # plot the crosstab
        pd.crosstab(df[feature],df[target]).plot(kind='bar', figsize=(20,5), stacked=True)
        plt.title(feature+' / '+target)
        plt.xlabel(feature)
        plt.ylabel(feature+' / '+target)

        plt.show()
            
        # display the table obove each chart 
        return num_vs_ctr(df, feature, target) 


def plot_cf_matrix_and_roc(model, X_train, y_train, X_test, y_test, y_pred, classes=[0,1], normalize=False, cmap=plt.cm.Blues):
        
    metrics_list = []
    
    # the main plot
    plt.figure(figsize=(15,5))

    # the confusion matrix
    plt.subplot(1,2,1)
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title("Normalized confusion matrix")
    else:
        plt.title('Confusion matrix')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # the result metrix
    summary_df = pd.DataFrame([[str(np.unique( y_pred )),
                            str(round(metrics.precision_score(y_test, y_pred.round()),3)),
                               str(round(metrics.accuracy_score(y_test, y_pred.round()),3)),
                               str(round(metrics.recall_score(y_test, y_pred.round(), average='binary'),3)),
                               str(round(metrics.roc_auc_score(y_test, y_pred.round()),3)),
                               str(round(metrics.cohen_kappa_score(y_test, y_pred.round()),3)),
                               str(round(metrics.f1_score(y_test, y_pred.round(), average='binary'),3))]], 
                              columns=['Class', 'Precision', 'Accuracy', 'Recall', 'ROC-AUC', 'Kappa', 'F1-score'])
    # print the metrics
    print("\n");
    print(summary_df);
    print("\n");
    
    plt.show()

    return



# analyzes the dataframe for data preparation
def exp_data_analysis(df, df_test):

    # divides the dataset as integer and object variables
    int_data = df.select_dtypes(include=[np.number])
    object_data = df.select_dtypes(include=[np.object])

    #print(int_data.head()) # Delete the "#" sign to run this line
    #print(object_data.head()) # Delete the "#" sign to run this line

    # summarizes the Train Data values
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Missing'] = df.isnull().sum().values   
    summary['Uniques'] = df.nunique().values 

    print("Summary Train Data Before Preprocessing") # Delete the "#" sign to run this line
    print(summary) # Delete the "#" sign to run this line

    # summarizes the dataset values
    summary_test = pd.DataFrame(df_test.dtypes, columns=['dtypes'])
    summary_test = summary_test.reset_index()
    summary_test['Missing'] = df_test.isnull().sum().values   
    summary_test['Uniques'] = df_test.nunique().values 

    print("Summary Test Data Before Preprocessing") # Delete the "#" sign to run this line
    print(summary_test) # Delete the "#" sign to run this line

    
    print(df['Response'].describe()) # Delete the "#" sign to run this line
    
    # plots the correlation heatmap
    corr = df.corr() # Delete the "#" sign to run this line
    mask = np.zeros_like(corr, dtype=np.bool) # Delete the "#" sign to run this line
    mask[np.triu_indices_from(mask)] = True # Delete the "#" sign to run this line
    sns.heatmap(corr, mask = mask, annot=True) # Delete the "#" sign to run this line
    plt.show()
    
    # City_Code column ivestigation
    print("City_Code Column Investigation Resuls") # Delete the "#" sign to run this line
    print("City_Code Column Description:") # Delete the "#" sign to run this line
    print(df['City_Code'].describe())  # Delete the "#" sign to run this line
    print("City_Code Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['City_Code'], 'Response')) # Delete the "#" sign to run this line
    print("City_Code Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['City_Code'].isnull().sum()) # Delete the "#" sign to run this line
    
    # replacing "C30","C31","C35" and "C36" labels with "C00" on training set
    df.loc[df['City_Code'] == 'C30', 'City_Code'] = 'C00'
    df.loc[df['City_Code'] == 'C31', 'City_Code'] = 'C00'
    df.loc[df['City_Code'] == 'C35', 'City_Code'] = 'C00'
    df.loc[df['City_Code'] == 'C36', 'City_Code'] = 'C00'

    # replacing "C30","C31","C35" and "C36" labels with "C00" on test set
    df_test.loc[df_test['City_Code'] == 'C30', 'City_Code'] = 'C00'
    df_test.loc[df_test['City_Code'] == 'C31', 'City_Code'] = 'C00'
    df_test.loc[df_test['City_Code'] == 'C35', 'City_Code'] = 'C00'
    df_test.loc[df_test['City_Code'] == 'C36', 'City_Code'] = 'C00'   

    print(crosstab(df, ['City_Code'], 'Response')) # Delete the "#" sign to run this line

    # Region_Code column investigation
    print("Region_Code Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Region_Code Column Description:") # Delete the "#" sign to run this line
    print(df['Region_Code'].describe())  # Delete the "#" sign to run this line
    print("Region_Code Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Region_Code'].isnull().sum()) # Delete the "#" sign to run this line
    sns.boxplot(x = 'Response', y = 'Region_Code', data = df) # Delete the "#" sign to run this line
    plt.show() # Delete the # Delete the "#" sign to run this line

    # Accomodation_Type column ivestigation
    print("Accomodation_Type Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Accomodation_Type Column Description:") # Delete the "#" sign to run this line
    print(df['Accomodation_Type'].describe())  # Delete the "#" sign to run this line
    print("Accomodation_Type Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['Accomodation_Type'], 'Response')) # Delete the "#" sign to run this line
    print("Accomodation_Type Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Accomodation_Type'].isnull().sum()) # Delete the "#" sign to run this line

    # convert to binary on trainingset
    df.loc[df['Accomodation_Type'] == 'Owned', 'Accomodation_Type'] = 1 
    df.loc[df['Accomodation_Type'] == 'Rented', 'Accomodation_Type'] = 0 

    # convert to int on trainingset
    df['Accomodation_Type'] = df['Accomodation_Type'].astype('int64') 

    # convert to binary on testset
    df_test.loc[df_test['Accomodation_Type'] == 'Owned', 'Accomodation_Type'] = 1 
    df_test.loc[df_test['Accomodation_Type'] == 'Rented', 'Accomodation_Type'] = 0 

    # convert to int on trainingset
    df_test['Accomodation_Type'] = df_test['Accomodation_Type'].astype('int64') 

    # Reco_Insurance_Type column ivestigation
    print("Reco_Insurance_Type Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Reco_Insurance_Type Column Description:") # Delete the "#" sign to run this line
    print(df['Reco_Insurance_Type'].describe())  # Delete the "#" sign to run this line
    print("Reco_Insurance_Type Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['Reco_Insurance_Type'], 'Response')) # Delete the "#" sign to run this line
    print("Reco_Insurance_Type Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Reco_Insurance_Type'].isnull().sum()) # Delete the "#" sign to run this line

    # convert to binary on trainingset
    df.loc[df['Reco_Insurance_Type'] == 'Individual', 'Reco_Insurance_Type'] = 1 
    df.loc[df['Reco_Insurance_Type'] == 'Joint', 'Reco_Insurance_Type'] = 0 

    # convert to int on trainingset
    df['Reco_Insurance_Type'] = df['Reco_Insurance_Type'].astype('int64')

    # convert to binary on testset
    df_test.loc[df_test['Reco_Insurance_Type'] == 'Individual', 'Reco_Insurance_Type'] = 1 
    df_test.loc[df_test['Reco_Insurance_Type'] == 'Joint', 'Reco_Insurance_Type'] = 0 

    # convert to int on testset
    df_test['Reco_Insurance_Type'] = df_test['Reco_Insurance_Type'].astype('int64')
    

    # Upper_Age column investigation
    print("Upper_Age Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Upper_Age Column Description:") # Delete the "#" sign to run this line
    print(df['Upper_Age'].describe())  # Delete the "#" sign to run this line
    print("Upper_Age Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Upper_Age'].isnull().sum()) # Delete the "#" sign to run this line
    sns.boxplot(x = 'Response', y = 'Upper_Age', data = df) # Delete the "#" sign to run this line
    plt.show() # Delete the # Delete the "#" sign to run this line

    # Lower_Age column investigation
    print("Lower_Age Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Lower_Age Column Description:") # Delete the "#" sign to run this line
    print(df['Lower_Age'].describe())  # Delete the "#" sign to run this line
    print("Lower_Age Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Lower_Age'].isnull().sum()) # Delete the "#" sign to run this line
    sns.boxplot(x = 'Response', y = 'Lower_Age', data = df) # Delete the "#" sign to run this line
    plt.show() # Delete the # Delete the "#" sign to run this line

    # Is_Spouse column ivestigation
    print("Is_Spouse Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Is_Spouse Column Description:") # Delete the "#" sign to run this line
    print(df['Is_Spouse'].describe())  # Delete the "#" sign to run this line
    print("Is_Spouse Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['Is_Spouse'], 'Response')) # Delete the "#" sign to run this line
    print("Is_Spousee Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Is_Spouse'].isnull().sum()) # Delete the "#" sign to run this line

        # convert to binary on trainingset
    df.loc[df['Is_Spouse'] == 'Yes', 'Is_Spouse'] = 1 
    df.loc[df['Is_Spouse'] == 'No', 'Is_Spouse'] = 0 

    # convert to int on trainingset
    df['Is_Spouse'] = df['Is_Spouse'].astype('int64')

    # convert to binary on testset
    df_test.loc[df_test['Is_Spouse'] == 'Yes', 'Is_Spouse'] = 1 
    df_test.loc[df_test['Is_Spouse'] == 'No', 'Is_Spouse'] = 0 

    # convert to int on testset
    df_test['Is_Spouse'] = df_test['Is_Spouse'].astype('int64')

    # Health Indicator column ivestigation
    print("Health Indicator Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Health Indicator Column Description:") # Delete the "#" sign to run this line
    print(df['Health Indicator'].describe())  # Delete the "#" sign to run this line
    print("Health Indicator Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['Health Indicator'], 'Response')) # Delete the "#" sign to run this line
    print("Health Indicator Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Health Indicator'].isnull().sum()) # Delete the "#" sign to run this line
    
    # replacing "X7","X8","X9" and "null" labels with "X0" on trainingset
    df['Health Indicator'].fillna('X0', inplace=True)

    df.loc[df['Health Indicator'] == 'X7', 'Health Indicator'] = 'X0'
    df.loc[df['Health Indicator'] == 'X8', 'Health Indicator'] = 'X0'
    df.loc[df['Health Indicator'] == 'X9', 'Health Indicator'] = 'X0'

    # replacing "X7","X8","X9" and "null" labels with "X0" on testset
    df_test['Health Indicator'].fillna('X0', inplace=True)

    df_test.loc[df_test['Health Indicator'] == 'X7', 'Health Indicator'] = 'X0'
    df_test.loc[df_test['Health Indicator'] == 'X8', 'Health Indicator'] = 'X0'
    df_test.loc[df_test['Health Indicator'] == 'X9', 'Health Indicator'] = 'X0'
    
    # Holding_Policy_Duration column ivestigation

    print("Holding_Policy_Duration Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Holding_Policy_Duration Column Description:") # Delete the "#" sign to run this line
    print(df['Holding_Policy_Duration'].describe())  # Delete the "#" sign to run this line
    print("Holding_Policy_Duration Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['Holding_Policy_Duration'], 'Response')) # Delete the "#" sign to run this line
    print("Holding_Policy_Duration Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Holding_Policy_Duration'].isnull().sum()) # Delete the "#" sign to run this line

    # replacing "null" labels with "other" on trainingset
    df['Holding_Policy_Duration'].fillna('Other', inplace=True)

    # replacing "null" labels with "other" on testset
    df_test['Holding_Policy_Duration'].fillna('Other', inplace=True)

    
    # Holding_Policy_Type column ivestigation

    print("Holding_Policy_Type Column Investigation Resuls") # Delete the "#" sign to run this line
    print("Holding_Policy_Type Column Description:") # Delete the "#" sign to run this line
    print(df['Holding_Policy_Type'].describe())  # Delete the "#" sign to run this line
    print("Holding_Policy_Type Column Values:") # Delete the "#" sign to run this line
    print(crosstab(df, ['Holding_Policy_Type'], 'Response')) # Delete the "#" sign to run this line
    print("Holding_Policy_Type Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Holding_Policy_Type'].isnull().sum()) # Delete the "#" sign to run this line

    # replacing "null" labels with "mean" value of the column on trainingset
    df['Holding_Policy_Type'].fillna(df['Holding_Policy_Type'].mean(), inplace=True)

    # replacing "null" labels with "mean" value of the column on testset
    df_test['Holding_Policy_Type'].fillna(df_test['Holding_Policy_Type'].mean(), inplace=True)


    # Reco_Policy_Cat column ivestigation
    
    print("Reco_Policy_Cat Investigation Resuls") # Delete the "#" sign to run this line
    print("Reco_Policy_Cat Description:") # Delete the "#" sign to run this line
    print(df['Reco_Policy_Cat'].describe())  # Delete the "#" sign to run this line
    print("Reco_Policy_Cat Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Reco_Policy_Cat'].isnull().sum()) # Delete the "#" sign to run this line
    sns.boxplot(x = 'Response', y = 'Reco_Policy_Cat', data = df) # Delete the "#" sign to run this line
    plt.show() # Delete the # Delete the "#" sign to run this line

    # outlier capping on trainingset
    Q1 = df['Reco_Policy_Cat'].quantile(0.95) 
    print("Total number of rows getting capped for Reco_Policy_Cat column : ",len(df[df['Reco_Policy_Cat'] >= Q1]))

    df.loc[df['Reco_Policy_Cat'] >= Q1, 'Reco_Policy_Cat'] = Q1 

    # outlier capping on testset
    Q2 = df_test['Reco_Policy_Cat'].quantile(0.95) 
    print("Total number of rows getting capped for Reco_Policy_Cat column : ",len(df_test[df_test['Reco_Policy_Cat'] >= Q2]))

    df_test.loc[df_test['Reco_Policy_Cat'] >= Q2, 'Reco_Policy_Cat'] = Q2 


    # Reco_Policy_Premium column ivestigation
    
    print("Reco_Policy_Premium Investigation Resuls") # Delete the "#" sign to run this line
    print("Reco_Policy_Premium Description:") # Delete the "#" sign to run this line
    print(df['Reco_Policy_Premium'].describe())  # Delete the "#" sign to run this line
    print("Reco_Policy_Premium Column Null Value Count:") # Delete the "#" sign to run this line
    print(df['Reco_Policy_Premium'].isnull().sum()) # Delete the "#" sign to run this line
    sns.boxplot(x = 'Response', y = 'Reco_Policy_Premium', data = df) # Delete the "#" sign to run this line
    plt.show() # Delete the # Delete the "#" sign to run this line

    # outlier capping on trainingset
    Q3 = df['Reco_Policy_Premium'].quantile(0.95) 
    print("Total number of rows getting capped for Reco_Policy_Premium column : ",len(df[df['Reco_Policy_Premium'] >= Q3]))

    df.loc[df['Reco_Policy_Premium'] >= Q3, 'Reco_Policy_Premium'] = Q3 

    # outlier capping on testset
    Q4 = df_test['Reco_Policy_Premium'].quantile(0.95) 
    print("Total number of rows getting capped for Reco_Policy_Premium column : ",len(df_test[df_test['Reco_Policy_Premium'] >= Q4]))

    df_test.loc[df_test['Reco_Policy_Premium'] >= Q4, 'Reco_Policy_Premium'] = Q4 

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Missing'] = df.isnull().sum().values   
    summary['Uniques'] = df.nunique().values 
    
    print("Summary After Preprocessin for Train Data") # Delete the "#" sign to run this line
    print(summary) # Delete the "#" sign to run this line

    summary_test = pd.DataFrame(df_test.dtypes, columns=['dtypes'])
    summary_test = summary_test.reset_index()
    summary_test['Missing'] = df_test.isnull().sum().values   
    summary_test['Uniques'] = df_test.nunique().values 
    
    print("Summary After Preprocessin for Test Data") # Delete the "#" sign to run this line
    print(summary_test) # Delete the "#" sign to run this line



    # Correlation after preprocessing
    sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(10,8)
    plt.show()

    return df, df_test


def build_model(df, df_test):

    # divides the dataframe as train and test sets.
    X = df.drop('Response', 1)
    y = df['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=47)
    X_train.shape, X_test.shape

    train_df = pd.concat([X_train, y_train], axis=1)

    #extracts categorical features
    cat_features=[i for i in X_train.columns if ((X_train.dtypes[i]!='int64') & (X_train.dtypes[i]!='float64'))]
    
    #extracts boolen features
    bool_features=[i for i in X_train.columns if (((X_train.dtypes[i]=='int64') | (X_train.dtypes[i]=='float64')) & (len(X_train[i].unique()) == 2))]

    #extracts numeric features
    num_features=[i for i in X_train.columns if (((X_train.dtypes[i]=='int64') | (X_train.dtypes[i]=='float64')) & (len(X_train[i].unique()) > 2))]
    print(num_features)


    from sklearn.utils import class_weight
    cw = list(class_weight.compute_class_weight('balanced',
                                             np.unique(y_train), y_train))
    cw = dict(enumerate(cw))
    
    params = {'depth':[2, 3, 4, 5],
            'iterations':[1500],
            'loss_function': ['Logloss'],
            'l2_leaf_reg':np.logspace(-19,-20,3),
            'early_stopping_rounds': [500],
            'learning_rate':[0.01],
            'eval_metric':['F1']
            }
    
    # pre-optimized parameters
    param = {'depth': 4,
        'early_stopping_rounds': 500,
        'eval_metric': 'AUC',
        'iterations': 2500,
        'l2_leaf_reg': 1e-19,
        'learning_rate': 0.01,
        'loss_function': 'Logloss'
    }


    #create the model
    clf2 = CatBoostClassifier(iterations=param['iterations'],
                            loss_function = param['loss_function'],
                            depth=param['depth'],
                            l2_leaf_reg = param['l2_leaf_reg'],
                            eval_metric = param['eval_metric'],
                            use_best_model=True,
                            early_stopping_rounds=param['early_stopping_rounds'],
                            class_weights = cw
    )

     #train the model
    clf2.fit(X_train, 
            y_train,
            cat_features=cat_features,
            logging_level='Silent',
            eval_set=(X_test, y_test)
    )
    

    feature_score = pd.DataFrame(list(zip(X_train.dtypes.index, clf2.get_feature_importance(Pool(X_train, label=train_df['Response'], cat_features=cat_features)))),
                    columns=['Feature','Score'])

    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
    plt.rcParams["figure.figsize"] = (15,8)
    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
    ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
    ax.set_xlabel('')

    rects = ax.patches

    labels = feature_score['Score'].round(2)

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='left', va='bottom')
    plt.xticks(rotation=85)

    plt.gca().invert_xaxis()

    plt.show()
    print(feature_score)

    pred_catboost2_train = clf2.predict(X_train)

    plot_cf_matrix_and_roc(clf2, X_train, y_train, X_train, y_train, pred_catboost2_train , classes=['NO','YES'])

    print(metrics.classification_report(y_train, pred_catboost2_train))

    # making predictions on test dataset
    
    ID = df_test['ID']
    df_test.drop(['ID'], inplace=True, axis=1)

    prediction = clf2.predict(df_test)

    
    submission_df = pd.DataFrame({
        'ID': ID,
        'Response': prediction,
    })
    submission_df.set_index('ID', inplace=True)
    print(submission_df.Response.value_counts())
    print(submission_df.head())
    submission_df.to_csv('submission.csv')


    return


def main():

    df, df_test = create_dataframe()
    df, df_test = exp_data_analysis(df, df_test)
    build_model(df, df_test)

    return

main()