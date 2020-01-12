import csv
import requests
from src import config

url = config.price_paid_url    

fname = config.project_home + "\\src\\data\\price_paid.csv"

def download_data(url = url, file_to_write = fname):
    print("Making get request from {}".format(url))
    response = requests.get(url, stream=True)
    print("Writing file to {}".format(file_to_write))
    print_after = 1000000
    i=0
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        for line in response.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))
            i+=1
            if i%print_after==0:
                print("Written {} lines...".format(i))
