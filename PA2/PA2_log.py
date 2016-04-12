#Machine Learning for Public Policy: HW1
#Charity King
#PA2: Logistic Regression Pipeline
#April 8, 2016
#csking1@uchicago.edu


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
TARGET = "Dlqin2yrs"
RENAME = ["Index", "Dlqin2yrs", "Unsecured", "Age", \
     "30to59days", "Debtratio", "Monthlyincome",  "Opencredit", \
    "90days","realestate", "60to89days", "Dependents"]

def ML_Pipeline(training, testing, file1, file2, png1, png2):
    '''
    Wrapper function that takes training/
    testing data
    '''
    print ("testing data stats")
    train = explore_data(training, png1)
    print ("unlabeled data stats")
    unlab = explore_data(testing, png2)
    filled = process_data(train, file1)
    filled_unlabeled = process_data(unlab, file2)
    train, test = train_test_split(filled, test_size = .1)
    model = build_model(train)
    test_model(test, model)
    classify_data(filled_unlabeled, model)
def build_model(dataframe):

    y = dataframe[TARGET]
    y = np.ravel(y)
    x = dataframe.drop(TARGET, 1)
    model = LogisticRegression()
    model= model.fit(x, y)
    print  ("Training Set with accuracy rating of:", model.score(x, y))
    return model
def test_model (dataframe, model): 
    y = dataframe[TARGET]
    y = np.ravel(y)
    x = dataframe.drop(TARGET, 1)
    testing = model.score(x, y)
    print ("Testing had a {} Accuracy Score". format(testing))
def classify_data(dataframe, model):

    y = dataframe[TARGET]
    y = np.ravel(y)
    x = dataframe.drop(TARGET, 1)
    predictions = model.predict(x)
    print (predictions)

def explore_data(dataframe, pngfile):
    '''
    Takes raw filename and 2 CSV filenames
    Calls conditional function to create two csv files
    '''
 
    df = pd.read_csv(dataframe)
    df.columns = RENAME
    df = df.drop("Index", 1)
    print_summary_stats(df)
    graph_histograms(df, pngfile)
    return df

def print_summary_stats(dataframe):
    '''
    Takes pandas dataframe and prints mean, median, mode, sd for each
    column
    '''
    
    # print (dataframe.count())
    # print (dataframe.describe())
    # print ("Median:", dataframe.median())
    # print ("Mode:", dataframe.mode())
    # print (dataframe.groupby("Dlqin2yrs").size())    
    # print (dataframe.groupby("Dlqin2yrs").mean())

    pass
def process_data(dataframe, filename):
    dataframe["Dependents"]= dataframe["Dependents"].fillna(0)
    filled_df = dataframe.fillna(dataframe.median())
    filename = filled_df.to_csv(filename, sep='\t')
    return filled_df
def create_bins():
    pass
def cat_to_binary():
    pass
    


def graph_histograms(dataframe, png_name):
    '''
    Takes dataframe and png_name and creates histogram png for dataset

    ["Index", "Dlqin2yrs", "Unsecured", "Age", \
     "30to59days", "Debtratio", "Monthlyincome",  "Opencredit", \
    "90days","realestate", "60to89days", "Dependents"]
    '''

    dataframe.Dlqin2yrs.hist(bins=4)
    plt.title("Histogram of Delinquency Risk")
    plt.xlabel('Dlqin Risk within 2 Years')
    plt.ylabel("Frequency")
    plt.savefig(png_name)



training = "cs-training.csv"
testing = "cs-test.csv"
file1 = "filledtraining.csv"
file2 = "filledtesting.csv"
png1 = "training.png"
png2 = "testing.png"

ML_Pipeline(training,testing, file1, file2, png1, png2)