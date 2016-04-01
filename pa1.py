import csv
import pandas as pd
from genderize import Genderize


def parse_data(filename):
    no_gender = []
    # print (Genderize().get(['James', 'Eva']))

    data_df = pd.read_csv(filename)
    print (data_df.columns)  #gives headers
    # print (data_df.describe())
    # print (data_df.std())
    if not data_df["Gender"]:
        l = data_df['First_name'].tolist()
        print (l)


    # with open(filename, 'rU') as f:
    #     fields = csv.reader(f)
    #     attrs = next(fields)
    #     for row in fields:
    #         print (row)




raw_filename = "mock_student_data.csv"
parse_data(raw_filename)