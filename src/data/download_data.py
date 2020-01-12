import csv
import requests
import logging
from pathlib import Path
from src import config

url = config.price_paid_url
    
fname = Path(config.project_home).joinpath("data","external","price_paid.csv")

def download_data(url = url, file_to_write = fname):
    if fname.exists():
        logging.warn("File already exists at {filename}. Data won't be downloaded".format(filename=str(fname)))
        return None
    logging.info("Making get request from {}".format(url))

    response = requests.get(url, stream=True)
    logging.info("Writing file to {}".format(file_to_write))
    print_after = 1000000
    i=0
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        for line in response.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))
            i+=1
            if i%print_after==0:
                logging.info("Written {} lines...".format(i))
