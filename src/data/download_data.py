import csv
import requests
from tqdm import tqdm
from src import config
import re

url = config.price_paid_url    

fname = config.project_home + "\\src\\data\\price_paid.csv"

def count_lines(iterator, max_lines):
    n_lines = 0
    while n_lines < max_lines:
        yield iterator.__next__()
        n_lines += 1

def download_data(url = url, file_to_write = fname, lines=None):
    print("Making get request from {}".format(url))
    response = requests.get(url, stream=True)
    print("Writing file to {}".format(file_to_write))

    if lines == None:
        response_iterator = tqdm(response.iter_lines())
    else:
        response_iterator = count_lines(tqdm(response.iter_lines()), lines)

    with open(fname, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for line in response_iterator:
            columns = re.findall(r'"(.*?)"', line.decode('utf-8'))
            writer.writerow(columns)

