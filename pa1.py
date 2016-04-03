import csv
import pandas as pd
from genderize import Genderize
import matplotlib.pyplot as plt


def parse_data(filename, file1):
 
    data_df = pd.read_csv(filename)
    # print ("This is the number of entries for each column")
    # print (data_df.count())  #gives headers
    print ("-----------------------------------------")
    new_df = data_df.fillna(data_df.mean())
    gender = new_df[new_df['Gender'].notnull()]
    no_gender = new_df[new_df['Gender'].isnull()]
    # for index, row in no_gender.iterrows():
    #     name = row["First_name"].split(" ")
    #     result =  Genderize().get(name)
    #     no_gender.is_copy = False  
    #     no_gender.loc[index, "Gender"] = result[0]["gender"]
    merged = no_gender.append(gender, ignore_index = True)
    print (merged.describe())
    print (merged.hist())
    merged.to_csv(file1, sep='\t')
    fig = plt.hist()
    merged.plot.hist()
    savefig('fig1.png')

    # print (data_df.describe())
    # print (data_df.std())





raw_filename = "mock_student_data.csv"
file1 = "filled_out.csv"
file2 = "conditional.csv"
file3 = "dropped.csv"
parse_data(raw_filename, file1)