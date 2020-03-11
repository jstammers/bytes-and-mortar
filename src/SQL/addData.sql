SET schema 'input';

create table PricePaid(tid VARCHAR, 
                       price FLOAT, 
                       transfer_date DATE, 
                       postcode VARCHAR, 
                       property_type CHAR(1), 
                       new_build CHAR(1), 
                       tenure CHAR(1), 
                       address1 VARCHAR,
                       address2 VARCHAR,
                       address3 VARCHAR,
                       address4 VARCHAR,
                       address5 VARCHAR,
                       address6 VARCHAR,
                       address7 VARCHAR ,
                       ppd_type CHAR, 
                       record_status CHAR);

COPY PricePaid(tid, price, transfer_date, postcode, property_type, new_build, tenure, address1, address2, address3, address4, address5, address6, address7, ppd_type, record_status)
FROM program 'cmd /c "type C:\DataScience\bytes-and-morter\src\data\price_paid.csv"' (DELIMITER ',', ENCODING 'utf-8', FORMAT CSV, NULL '');