cp /postgresql.conf /var/lib/postgresql/data/postgresql.conf
cat /imdb_backup/imdb_pg11 | pg_restore -U postgres -d imdb