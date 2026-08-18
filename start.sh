#!/bin/bash

# ECG Classification System - Startup Script
# This script starts the backend server and opens the UI in browser

set -e  # Exit on error

# ============================================================================
# COLORS
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ECG Cardiac Block Detection System - Startup Script    ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# CHECKS
# ============================================================================

print_header

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_info "Python 3 found: $(python3 --version)"

# Check dependencies
echo ""
echo "Checking dependencies..."

DEPS_OK=true

if ! python3 -c "import flask" 2>/dev/null; then
    print_warning "Flask not found. Installing..."
    pip3 install flask flask-cors
else
    print_info "Flask installed"
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    print_warning "NumPy not found. Installing..."
    pip3 install numpy
else
    print_info "NumPy installed"
fi

if ! python3 -c "import torch" 2>/dev/null; then
    print_warning "PyTorch not found. Installing..."
    pip3 install torch
else
    print_info "PyTorch installed"
fi

if ! python3 -c "import tensorflow" 2>/dev/null; then
    print_warning "TensorFlow not found. Installing..."
    pip3 install tensorflow
else
    print_info "TensorFlow installed"
fi

# ============================================================================
# TEST PIPELINE
# ============================================================================

echo ""
echo "Running pipeline tests..."

if python3 test_pipeline.py > /tmp/test_output.log 2>&1; then
    print_info "All pipeline tests passed ✓"
else
    print_warning "Some tests failed (see test output)"
    tail -20 /tmp/test_output.log
fi

# ============================================================================
# START SERVER
# ============================================================================

echo ""
echo "Starting ECG Classification Backend..."
echo ""

PORT=5000
HOST="127.0.0.1"

# Check if port is already in use
if lsof -i :$PORT &> /dev/null; then
    print_warning "Port $PORT is already in use"
    read -p "Use a different port? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter port number: " PORT
    fi
fi

print_info "Starting server on http://$HOST:$PORT"
echo ""
echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
echo "Server is running. Press Ctrl+C to stop."
echo "Open browser: http://$HOST:$PORT"
echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
echo ""

# Open browser automatically
sleep 2
if command -v open &> /dev/null; then
    # macOS
    open "http://$HOST:$PORT"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "http://$HOST:$PORT"
elif command -v start &> /dev/null; then
    # Windows
    start "http://$HOST:$PORT"
fi

# Start server
export FLASK_PORT=$PORT
export FLASK_HOST=$HOST
python3 app.py
