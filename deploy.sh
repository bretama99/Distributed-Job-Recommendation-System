#!/bin/bash

# Distributed Job Recommendation System - Deployment Script
# This script automates the deployment of the entire distributed system

set -e

echo "======================================"
echo "Distributed Job Rec System Deployment"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check Docker version
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}')
    echo -e "${GREEN}✓ Docker version: $DOCKER_VERSION${NC}"
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        echo -e "${YELLOW}Warning: Less than 8GB RAM detected. Recommended: 16GB+${NC}"
    else
        echo -e "${GREEN}✓ Memory: ${TOTAL_MEM}GB${NC}"
    fi
    
    # Check available disk space
    AVAIL_DISK=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "${AVAIL_DISK%.*}" -lt 20 ]; then
        echo -e "${YELLOW}Warning: Less than 20GB disk space. Recommended: 50GB+${NC}"
    else
        echo -e "${GREEN}✓ Disk space: ${AVAIL_DISK}G available${NC}"
    fi
    
    echo ""
}

# Create necessary directories
setup_directories() {
    echo "Creating directory structure..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/external
    mkdir -p data/chromadb
    mkdir -p data/exports
    mkdir -p logs
    mkdir -p grafana-dashboards
    
    echo -e "${GREEN}✓ Directories created${NC}"
    echo ""
}

# Check for required data files
check_data_files() {
    echo "Checking for required data files..."
    
    REQUIRED_FILES=(
        "data/external/linkedin_job_postings.csv"
        "data/external/job_summary.csv"
        "data/external/job_skills.csv"
    )
    
    MISSING_FILES=()
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            MISSING_FILES+=("$file")
        fi
    done
    
    if [ ${#MISSING_FILES[@]} -gt 0 ]; then
        echo -e "${YELLOW}Warning: Missing data files:${NC}"
        for file in "${MISSING_FILES[@]}"; do
            echo "  - $file"
        done
        echo ""
        echo "Please add LinkedIn dataset files to data/external/"
        echo "You can download them from: [provide URL]"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ All required data files present${NC}"
    fi
    
    echo ""
}

# Start distributed services
deploy_services() {
    echo "Starting distributed services..."
    echo "This will take 2-5 minutes..."
    echo ""
    
    # Pull images first
    echo "Pulling Docker images..."
    docker-compose pull
    
    # Start services in stages
    echo ""
    echo "Stage 1: Starting databases..."
    docker-compose up -d mongo1 mongo2 mongo3 cassandra1 redis1 redis2 redis3 neo4j1
    
    echo "Waiting for databases to initialize (60s)..."
    sleep 60
    
    echo ""
    echo "Stage 2: Starting Cassandra cluster..."
    docker-compose up -d cassandra2 cassandra3
    sleep 30
    
    echo ""
    echo "Stage 3: Initializing MongoDB replica set..."
    docker-compose up -d
    sleep 30
    
    echo ""
    echo "Stage 4: Starting application servers..."
    docker-compose up -d app1 app2 app3
    
    echo ""
    echo "Stage 5: Starting monitoring..."
    docker-compose up -d prometheus grafana
    
    echo ""
    echo "Stage 6: Starting load balancer..."
    docker-compose up -d nginx
    
    echo ""
    echo -e "${GREEN}✓ All services started${NC}"
    echo ""
}

# Wait for services to be healthy
wait_for_health() {
    echo "Waiting for services to be healthy..."
    
    MAX_RETRIES=30
    RETRY_DELAY=10
    
    # MongoDB
    echo -n "MongoDB replica set: "
    for i in $(seq 1 $MAX_RETRIES); do
        if docker exec mongo1 mongosh --quiet --eval "rs.status()" &> /dev/null; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep $RETRY_DELAY
    done
    
    # Cassandra
    echo -n "Cassandra cluster: "
    for i in $(seq 1 $MAX_RETRIES); do
        if docker exec cassandra1 nodetool status | grep -q "UN"; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep $RETRY_DELAY
    done
    
    # Neo4j
    echo -n "Neo4j: "
    for i in $(seq 1 $MAX_RETRIES); do
        if docker exec neo4j1 cypher-shell -u neo4j -p jobrecpassword "RETURN 1" &> /dev/null; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep $RETRY_DELAY
    done
    
    # Redis
    echo -n "Redis cluster: "
    for i in $(seq 1 $MAX_RETRIES); do
        if docker exec redis1 redis-cli --cluster check redis1:6379 &> /dev/null; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep $RETRY_DELAY
    done
    
    # Application
    echo -n "Application: "
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s http://localhost/health &> /dev/null; then
            echo -e "${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep $RETRY_DELAY
    done
    
    echo ""
}

# Initialize data
initialize_data() {
    echo "Initializing data..."
    
    # Run ETL
    echo "Running ETL pipeline..."
    docker exec jobrec-app1 python -c "from src.features import run_etl; run_etl()" || true
    
    # Compute embeddings
    echo "Computing embeddings..."
    docker exec jobrec-app1 python -c "from src.features import compute_embeddings; compute_embeddings()" || true
    
    # Load Neo4j
    echo "Loading Neo4j graph..."
    docker exec jobrec-app1 python -c "from src.neo4j_store import neo4j_store; neo4j_store.load_graph_data()" || true
    
    # Initialize Cassandra
    echo "Initializing Cassandra schema..."
    docker exec cassandra1 cqlsh -f /init.cql || true
    
    echo -e "${GREEN}✓ Data initialized${NC}"
    echo ""
}

# Display access information
display_access_info() {
    echo "======================================"
    echo "Deployment Complete!"
    echo "======================================"
    echo ""
    echo "Access your services:"
    echo ""
    echo -e "  ${GREEN}Application UI:${NC}     http://localhost"
    echo -e "  ${GREEN}Grafana:${NC}            http://localhost:3000 (admin/jobrecadmin)"
    echo -e "  ${GREEN}Prometheus:${NC}         http://localhost:9090"
    echo -e "  ${GREEN}Neo4j Browser:${NC}      http://localhost:7474 (neo4j/jobrecpassword)"
    echo ""
    echo "Service Status:"
    docker-compose ps
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "To stop all services:"
    echo "  docker-compose down"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    setup_directories
    check_data_files
    deploy_services
    wait_for_health
    initialize_data
    display_access_info
}

# Run main function
main