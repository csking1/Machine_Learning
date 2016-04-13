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


def model_switch(training, testing, file1, file2, file3, png1, png2):
    print ("First model using Continuous Age variable")
    ML_Pipeline(training, testing, file1, file2, file3, png1, png2, discretized=False)
    print ("Second Model using Discretized Age Variable")
 
    file1 = "dis_filledtraining.csv"
    file2 = "dis_filledtesting.csv"
    file3 = "dis_classifiedtesting.csv"
    png1 = "dis_training.png"
    png2 = "dis_testing.png"

    ML_Pipeline(training, testing, file1, file2, file3, png1, png2, True)

def ML_Pipeline(training, testing, file1, file2, file3, png1, png2, discretized):
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
    if discretized:
        filled=create_bins(filled, 'Age')
        filled_unlabeled = create_bins(filled_unlabeled, 'Age')
    train, test = train_test_split(filled, test_size = .1)
    model = build_model(train)
    test_model(test, model)
    classify_data(filled_unlabeled, model, file3)

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
def classify_data(dataframe, model, filename):

    y = dataframe[TARGET]
    y = np.ravel(y)
    x = dataframe.drop(TARGET, 1)
    predictions = model.predict(x)
    p = pd.DataFrame(predictions)
    p.rename(columns={p.columns[0]: "Dlqin2yrs"}, inplace=True)
    result = pd.concat([p, x], axis=1, join_axes=[p.index])
    filename = result.to_csv(filename, sep='\t')
    print ("classifed histogram")
    graph_histograms(result, "classified.png")
    print (p.groupby("Dlqin2yrs").size())    

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
    # print ("stats for y var.......")
    # print (dataframe.groupby("Dlqin2yrs").size())    
    # print (dataframe.groupby("Dlqin2yrs").mean())

    pass
def process_data(dataframe, filename):
    dataframe["Dependents"]= dataframe["Dependents"].fillna(0)
    filled_df = dataframe.fillna(dataframe.median())
    filename = filled_df.to_csv(filename, sep='\t')
    return filled_df
def create_bins(dataframe, variable):
    '''
    Takes a dataframe and the Age variable
    Creates four discretized bins
    '''
    dataframe['Age20-30'] = dataframe.apply \
    (lambda row: 1 if ((row[variable] >= 20 and row[variable] < 30)) \
        else 0, axis=1)
    dataframe['Age30-40'] = dataframe.apply \
    (lambda row: 1 if ((row[variable] >= 30 and row[variable] < 40)) \
        else 0, axis=1)
    dataframe['Age40-60'] = dataframe.apply \
    (lambda row: 1 if ((row[variable] >= 40 and row[variable] < 60)) \
        else 0, axis=1)
    dataframe['Age60+'] = dataframe.apply \
    (lambda row: 1 if ((row[variable] >= 60 and row[variable] < 120)) \
        else 0, axis=1)

    return dataframe


def graph_histograms(dataframe, png_name):
    '''
    Takes dataframe and png_name and creates histogram png for dataset

    ["Index", "Dlqin2yrs", "Unsecured", "Age", \
     "30to59days", "Debtratio", "Mont
     hlyincome",  "Opencredit", \
    "90days","realestate", "60to89days", "Dependents"]
    '''

    dataframe.Dlqin2yrs.hist(bins=4)
    plt.title("Histogram of Delinquency Risk")
    plt.xlabel('Dlqin Risk within 2 Years')
    plt.ylabel("Frequency")
    plt.savefig(png_name)
    #---------------------------------------------

    dataframe.hist()
    # plt.title("Histogram of Delinquency Risk")
    # plt.xlabel('Dlqin Risk within 2 Years')
    # plt.ylabel("Frequency")
    plt.savefig("ugly.png")

training = "cs-training.csv"
testing = "cs-test.csv"
file1 = "filledtraining.csv"
file2 = "filledtesting.csv"
file3 = "classifiedtesting.csv"
png1 = "training.png"
png2 = "testing.png"

model_switch(training,testing, file1, file2, file3, png1, png2)