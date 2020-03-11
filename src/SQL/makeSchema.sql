create user bytes_and_mortar;
create schema if not exists input;
grant all on schema input to bytes_and_mortar;
grant all on all tables in schema input to bytes_and_mortar;