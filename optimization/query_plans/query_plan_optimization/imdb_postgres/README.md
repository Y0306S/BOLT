# PostgreSQL Image with IMDB Dataset

NOTE: This is for testing purposes only. For real performance measurements, you should probably deploy a real cluster.

# Setup

1. Download https://doi.org/10.7910/DVN/2QYZBT into this folder
2. `docker build -t imdb-postgres:latest .`
3. `docker run --name=imdb imdb-postgres`
4. Wait for a while (takes ~3 minutes on my machine)
5. The log line `database system is ready to accept connections` means the DB is ready
