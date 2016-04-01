import csv

def parse_data(filename):

    with open(filename, 'rU') as f:
        fields = csv.reader(f)
        attrs = next(fields)
        for row in fields:
            print (row)




raw_filename = "mock_student_data.csv"
parse_data(raw_filename)