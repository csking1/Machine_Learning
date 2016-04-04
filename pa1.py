#Machine Learning for Public Policy: HW1
#Charity King
#April 4, 2016
#csking1@uchicago.edu


import pandas as pd
from genderize import Genderize
import matplotlib.pyplot as plt


def parse_data(filename, file1, file2):
    '''
    Takes raw filename and 2 CSV filenames
    Calls conditional function to create two csv files
    '''
 
    data_df = pd.read_csv(filename)
    print_summary_stats(data_df)
    gender = data_df[data_df['Gender'].notnull()]
    no_gender = data_df[data_df['Gender'].isnull()]
    for index, row in no_gender.iterrows():
        name = row["First_name"].split(" ")
        result =  Genderize().get(name)
        no_gender.is_copy = False  
        no_gender.loc[index, "Gender"] = result[0]["gender"]
    merged = no_gender.append(gender, ignore_index = True)
    conditional(merged, "uncond_hist.png", file1, conditional=False)
    conditional(merged,"condit_hist.png", file2, conditional=True)

def print_summary_stats(dataframe):
    '''
    Takes pandas dataframe and prints mean, median, mode, sd for each
    column
    '''
    print ("------------Summary Statistics-------------")
    print (dataframe.describe())
    print ("Median:", dataframe.median())
    print ("Mode:", dataframe.mode())

def conditional(dataframe, png_name, file, conditional):
    '''
    Takes dataframe, png name, csv filename, and conditional switch
    Creates a csv file after filling missing data conditional or unconditional
    Calls histogram graphing function
    '''
    if not conditional:
        new_dataframe = dataframe.fillna(dataframe.mean())
        new_dataframe.to_csv(file, sep='\t')
        graph_histograms(new_dataframe, png_name)
    else:
        print ("--------in conditional function-------------")
        graduated = dataframe[dataframe.Graduated == "Yes"]
        not_grad = dataframe[dataframe.Graduated == "No"]
        print ("number of entries for non-grads", not_grad.count())
        print ("number of entries for grad", graduated.count())
        graduated1 = graduated.fillna(graduated.mean())
        total = graduated1.append(not_grad, ignore_index=True)
        total.to_csv(file, sep='\t')
        graph_histograms(total, png_name)

def graph_histograms(dataframe, png_name):
    '''
    Takes dataframe and png_name and creates histogram png for dataset
    '''

    dataframe.hist()
    plt.savefig(png_name)
    print_summary_stats(dataframe)



raw_filename = "mock_student_data.csv"
file1 = "uncondit.csv"
file2 = "condit.csv"
file3 = "dropped.csv"
parse_data(raw_filename, file1, file2)