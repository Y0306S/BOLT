# Parallel Database Setup for LOLBO Optimization

## Overview

This guide explains how to set up multiple PostgreSQL instances to parallelize query evaluation in LOLBO, reducing optimization time from ~7 days to ~40-50 hours.

## Why Parallel Databases?

In sequential mode (batch size = 1):
- Each BO iteration evaluates 1 query plan
- 100 configurations = 100 sequential evaluations
- Estimated time: ~8-10 minutes per task × 1000 tasks = **~166 hours (7 days)**

With 4 parallel databases (batch size = 4):
- Each BO iteration evaluates 4 query plans simultaneously
- 100 configurations = 25 batched iterations
- Estimated time: ~8-10 minutes per batch × 250 batches = **~42 hours**

**Speedup: ~4x faster!**

## Quick Start

### Step 1: Setup Multiple PostgreSQL Instances

```bash
cd /workspace/optimization/query_plans/query_plan_optimization/lolbo_scripts

# Make script executable
chmod +x setup_parallel_databases.sh

# Setup 4 PostgreSQL instances (default)
./setup_parallel_databases.sh

# Or specify number of instances
./setup_parallel_databases.sh 4
```

This creates instances on ports:
- Instance 1: localhost:5433
- Instance 2: localhost:5434
- Instance 3: localhost:5435
- Instance 4: localhost:5436

### Step 2: Start All Instances

```bash
# Start each instance
pg_ctl -D $HOME/postgres_parallel/instance_1 -l $HOME/postgres_parallel/instance_1.log start
pg_ctl -D $HOME/postgres_parallel/instance_2 -l $HOME/postgres_parallel/instance_2.log start
pg_ctl -D $HOME/postgres_parallel/instance_3 -l $HOME/postgres_parallel/instance_3.log start
pg_ctl -D $HOME/postgres_parallel/instance_4 -l $HOME/postgres_parallel/instance_4.log start
```

### Step 3: Load IMDB Schema into Each Instance

```bash
# Load schema and data into all instances
for i in $(seq 1 4); do
    port=$((5432 + $i))
    echo "Loading IMDB into instance on port $port..."
    psql -h localhost -p $port -d postgres -f /path/to/imdb_schema.sql
    psql -h localhost -p $port -d postgres -f /path/to/imdb_data.sql
done
```

### Step 4: Run Parallel Optimization

```bash
# Run with 4 parallel database instances
python run_CEB_64_parallel_100tasks.py \
    --workload_name CEB_1A10 \
    --vae_ckpt_path ../vae/CEB_64.ckpt \
    --wandb_entity your_username \
    --num_db_instances 4
```

## Configuration Details

### Database Configuration (Optimized for Query Plans)

Each instance is configured with:
```conf
shared_buffers = 256MB           # Memory for caching
effective_cache_size = 1GB       # OS cache estimate
work_mem = 16MB                  # Per-operation memory
maintenance_work_mem = 128MB     # For VACUUM, CREATE INDEX
random_page_cost = 1.1           # SSD optimization
effective_io_concurrency = 200   # SSD parallel I/O
max_parallel_workers_per_gather = 2
max_parallel_workers = 4
```

### Python Script Configuration

Key parameters in `run_CEB_64_parallel_100tasks.py`:
```python
db_ports=[5433, 5434, 5435, 5436]  # Multiple DB instances
bsz=4                               # Batch size = 4 (parallel evaluation)
num_initialization_points=10        # 10 init points
max_n_oracle_calls=100              # Total: 10 init + 90 BO = 100 configs
```

## Manual Setup (Alternative)

If you prefer manual setup or Docker:

### Option A: Using Docker

```bash
# Create 4 Docker containers
for i in $(seq 1 4); do
    port=$((5432 + $i))
    docker run -d \
        --name postgres-lolbo-$i \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_USER=imdb \
        -e POSTGRES_DB=imdb \
        -p $port:5432 \
        -v /data/postgres_$i:/var/lib/postgresql/data \
        postgres:14
done

# Load schema into each container
for i in $(seq 1 4); do
    port=$((5432 + $i))
    docker exec -i postgres-lolbo-$i psql -U imdb -d imdb < imdb_schema.sql
    docker exec -i postgres-lolbo-$i psql -U imdb -d imdb < imdb_data.sql
done
```

### Option B: System PostgreSQL (Multiple Clusters)

```bash
# Create multiple clusters (Debian/Ubuntu)
for i in $(seq 1 4); do
    port=$((5432 + $i))
    pg_createcluster $PG_VERSION cluster_$i --port $port
    pg_ctlcluster $PG_VERSION cluster_$i start
done

# Load data into each cluster
for i in $(seq 1 4); do
    port=$((5432 + $i))
    psql -h localhost -p $port -U postgres -f imdb_schema.sql
    psql -h localhost -p $port -U postgres -f imdb_data.sql
done
```

## Verification

Check that all instances are running:

```bash
# Test connections to all instances
for i in $(seq 1 4); do
    port=$((5432 + $i))
    echo "Testing port $port..."
    psql -h localhost -p $port -d postgres -c "SELECT version();"
done
```

Expected output: 4 successful connections showing PostgreSQL version.

## Performance Tuning

### For NVMe SSD (Recommended)
```conf
random_page_cost = 1.1
effective_io_concurrency = 200
```

### For HDD
```conf
random_page_cost = 4.0
effective_io_concurrency = 2
```

### Increase Memory (if available)
```conf
shared_buffers = 512MB      # Up to 25% of RAM
effective_cache_size = 2GB  # Up to 50-75% of RAM
work_mem = 32MB             # Per operation
```

### Query Timeout Settings
To prevent hanging queries:
```sql
-- Set statement timeout (in milliseconds)
ALTER DATABASE imdb SET statement_timeout = '300000';  -- 5 minutes
```

## Monitoring

Monitor resource usage during optimization:

```bash
# Check CPU and memory usage
htop

# Check disk I/O
iotop

# Check PostgreSQL activity
for i in $(seq 1 4); do
    port=$((5432 + $i))
    psql -h localhost -p $port -d imdb -c "SELECT count(*) FROM pg_stat_activity;"
done
```

## Stopping Instances

```bash
# Stop all instances gracefully
for i in $(seq 1 4); do
    pg_ctl -D $HOME/postgres_parallel/instance_$i stop
done

# Or force stop
for i in $(seq 1 4); do
    pg_ctl -D $HOME/postgres_parallel/instance_$i stop -m immediate
done
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using the port
lsof -i :5433

# Kill the process if needed
kill -9 <PID>
```

### Connection Refused
```bash
# Check if instance is running
pg_ctl -D $HOME/postgres_parallel/instance_1 status

# Check logs
cat $HOME/postgres_parallel/instance_1.log
```

### Out of Memory
Reduce `shared_buffers` and `work_mem` in postgresql.conf for each instance.

### Slow Queries
1. Ensure statistics are up to date:
   ```sql
   ANALYZE;
   ```
2. Check query plans:
   ```sql
   EXPLAIN ANALYZE <your_query>;
   ```

## Expected Performance

| Configuration | Time per 100 configs | Time for 1000 tasks | Speedup |
|--------------|---------------------|-------------------|---------|
| Sequential (1 DB) | ~8-10 min | ~166 hours (7 days) | 1x |
| 2 Parallel DBs | ~4-5 min | ~83 hours | 2x |
| 4 Parallel DBs | ~2-2.5 min | ~42 hours | 4x |
| 8 Parallel DBs | ~1-1.25 min | ~21 hours | 8x |

*Note: Actual times depend on query complexity, hardware, and database configuration.*

## Advanced: Custom Number of Instances

To use a different number of parallel instances:

```bash
# Setup N instances
./setup_parallel_databases.sh 8

# Run optimization with 8 parallel DBs
python run_CEB_64_parallel_100tasks.py \
    --workload_name CEB_1A10 \
    --vae_ckpt_path ../vae/CEB_64.ckpt \
    --num_db_instances 8
```

## Next Steps

1. ✅ Setup 4 PostgreSQL instances
2. ✅ Load IMDB schema into each instance
3. ✅ Upload your VAE checkpoint (`CEB_64.ckpt`)
4. ✅ Verify initialization data exists
5. ✅ Run parallel optimization
6. 📊 Monitor progress via wandb dashboard

For more details, see:
- `README_LOLBO_SETUP.md` - General LOLBO setup guide
- `SETUP_SUMMARY.md` - Quick reference summary
- `run_CEB_64_parallel_100tasks.py` - Parallel optimization script
