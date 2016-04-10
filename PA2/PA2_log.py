#Machine Learning for Public Policy: HW1
#Charity King
#PA2: Logistic Regression Pipeline
#April 8, 2016
#csking1@uchicago.edu


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
TARGET = "Dlqin2yrs"
RENAME = ["Index", "Dlqin2yrs", "Unsecured", "Age", \
     "30to59days", "Debtratio", "Monthlyincome",  "Opencredit", \
    "90days","realestate", "60to89days", "Dependents"]

def ML_Pipeline(training, testing):
    '''
    Wrapper function that takes training/
    testing data
    '''
    train = explore_data(training)
    filled = process_data(train)
    build_model(filled)

def build_model(dataframe):

    y = dataframe[TARGET]
    y = np.ravel(y)
    x = dataframe.drop(TARGET, 1)
    model = LogisticRegression()
    model= model.fit(x, y)
    print (model.score(x, y))







def explore_data(training):
    '''
    Takes raw filename and 2 CSV filenames
    Calls conditional function to create two csv files
    '''
 
    train = pd.read_csv(training)
    train.columns = RENAME
    # print (train["Dlqin2yrs"])
    print_summary_stats(train)
    graph_histograms(train, "initial.png")
    return train
    # gender = data_df[data_df['Gender'].notnull()]
    # no_gender = data_df[data_df['Gender'].isnull()]
    # for index, row in no_gender.iterrows():
    #     name = row["First_name"].split(" ")
    #     result =  Genderize().get(name)
    #     no_gender.is_copy = False  
    #     no_gender.loc[index, "Gender"] = result[0]["gender"]
    # merged = no_gender.append(gender, ignore_index = True)
    # conditional(merged, "uncond_hist.png", file1, conditional=False)
    # conditional(merged,"condit_hist.png", file2, conditional=True)

def print_summary_stats(dataframe):
    '''
    Takes pandas dataframe and prints mean, median, mode, sd for each
    column
    '''
    pass
    # print (dataframe.count())
    # print (dataframe.describe())
    # print ("Median:", dataframe.median())
    # print ("Mode:", dataframe.mode())
    # print (dataframe.groupby("Dlqin2yrs").size())    
    # print (dataframe.groupby("Dlqin2yrs").mean())


def process_data(dataframe):
    # print ("preposcessing data-----------------")
    filled = dataframe.fillna(dataframe.median())
    #potentially categorize
    return filled
    


def graph_histograms(dataframe, png_name):
    '''
    Takes dataframe and png_name and creates histogram png for dataset

    ["Index", "Dlqin2yrs", "Unsecured", "Age", \
     "30to59days", "Debtratio", "Monthlyincome",  "Opencredit", \
    "90days","realestate", "60to89days", "Dependents"]
    '''

    dataframe.Dlqin2yrs.hist(xlabelsize=2)
    plt.title("Histogram of Delinquency Risk")
    plt.xlabel('Dlqin Risk within 2 Years')
    plt.ylabel("Frequency")
    plt.savefig(png_name)



training = "cs-training.csv"
testing = "cs-test.csv"
file1 = "uncondit.csv"
file2 = "condit.csv"

ML_Pipeline(training,testing)