"%POSTGRES_HOME%\psql" -U postgres -f "%~dp0src\SQL\makeSchema.sql"
"%POSTGRES_HOME%\psql" -U postgres -f "%~dp0src\SQL\addData.sql"