import src
import os

src_path, src_filename = os.path.split(src.__file__)
project_home = os.path.dirname(src_path)

price_paid_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv"