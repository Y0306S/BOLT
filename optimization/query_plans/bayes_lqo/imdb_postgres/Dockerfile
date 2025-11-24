FROM postgres:16-bookworm

ENV POSTGRES_PASSWORD=postgres

COPY imdb_pg11 /imdb_backup/
COPY postgresql.conf /postgresql.conf
COPY 0_create_imdb.sql /docker-entrypoint-initdb.d/
COPY 1_restore.sh /docker-entrypoint-initdb.d/

RUN apt update && apt install -y postgresql-common ca-certificates
RUN sh /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -i -v 16
RUN apt install -y postgresql-16-pg-hint-plan