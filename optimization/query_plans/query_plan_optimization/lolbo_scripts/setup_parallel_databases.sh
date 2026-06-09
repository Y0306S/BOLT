#!/bin/bash
# Setup Multiple PostgreSQL Instances for Parallel LOLBO Optimization
# This script creates 4 PostgreSQL instances on different ports for parallel query evaluation

set -e

NUM_INSTANCES=${1:-4}
BASE_PORT=5432
DATA_DIR="${HOME}/postgres_parallel"
PG_VERSION=${2:-14}

echo "Setting up $NUM_INSTANCES PostgreSQL instances..."

# Create base directory
mkdir -p "$DATA_DIR"

# Function to setup a single instance
setup_instance() {
    local instance_id=$1
    local port=$((BASE_PORT + instance_id))
    local instance_dir="$DATA_DIR/instance_$instance_id"
    
    echo "Setting up instance $instance_id on port $port..."
    
    # Create data directory
    mkdir -p "$instance_dir"
    
    # Initialize database cluster
    if [ ! -f "$instance_dir/PG_VERSION" ]; then
        initdb -D "$instance_dir" --encoding=UTF8 --locale=C
    fi
    
    # Configure postgresql.conf for this instance
    cat > "$instance_dir/postgresql.conf" << EOF
# Instance $instance_id Configuration
listen_addresses = 'localhost'
port = $port
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 128MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
max_worker_processes = 4
max_parallel_workers_per_gather = 2
max_parallel_workers = 4
max_parallel_maintenance_workers = 2
logging_collector = off
EOF
    
    # Configure pg_hba.conf for local connections
    cat > "$instance_dir/pg_hba.conf" << EOF
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
EOF
    
    echo "Instance $instance_id configured on port $port"
}

# Setup all instances
for i in $(seq 1 $NUM_INSTANCES); do
    setup_instance $i
done

echo ""
echo "All instances configured. To start them, run:"
echo ""
for i in $(seq 1 $NUM_INSTANCES); do
    port=$((BASE_PORT + i))
    instance_dir="$DATA_DIR/instance_$i"
    echo "pg_ctl -D $instance_dir -l $DATA_DIR/instance_$i.log start"
done

echo ""
echo "To load IMDB schema into each instance:"
echo "for i in \$(seq 1 $NUM_INSTANCES); do"
echo "    port=\$((5432 + \$i))"
echo "    psql -h localhost -p \$port -d postgres -f /path/to/imdb_schema.sql"
echo "    psql -h localhost -p \$port -d postgres -f /path/to/imdb_data.sql"
echo "done"
echo ""
echo "To stop all instances:"
echo "for i in \$(seq 1 $NUM_INSTANCES); do"
echo "    pg_ctl -D $DATA_DIR/instance_\$i stop"
echo "done"
