import csv
import requests
import logging
from pathlib import Path
from src import config

url = config.price_paid_url

fname = Path(config.project_home).joinpath("data","external","price_paid.csv")

def check_file_exists(filename: Path) -> bool:
    if fname.exists():
        logging.warn("File already exists at {filename}. Data won't be downloaded".format(filename=str(fname)))
        return True
    else: 
        return False

def download_data(url = url, file_to_write = fname):
    print_after = 1000000
    if check_file_exists(file_to_write):
        return None
    logging.info("Making get request from {}".format(url))
    response = requests.get(url, stream=True)
    logging.info("Writing file to {}".format(file_to_write))
    stream_to_file(response, file_to_write, print_after)


def stream_to_file(response, file_to_write, line_print_num):
    with open(file_to_write, 'w') as f:
        writer = csv.writer(f)
        for i,line in enumerate(response.iter_lines()):
            writer.writerow(line.decode('utf-8').split(','))
            if i%line_print_num==0:
                logging.info("Written {} lines...".format(i))

