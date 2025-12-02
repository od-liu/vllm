#!/bin/bash
# Quick script to run benchmark with custom timeout settings

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TIMEOUT=""
CONFIG_FILE=""
PHASE=2
GPUS=""

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run vLLM benchmark with custom timeout settings"
    echo ""
    echo "Options:"
    echo "  -t, --timeout SECONDS    Set timeout in seconds (or 'none' to disable)"
    echo "  -c, --config FILE        Config file path (required)"
    echo "  -p, --phase PHASE        Phase number (1 or 2, default: 2)"
    echo "  -g, --gpus IDS           GPU IDs (e.g., '0,1,2,3')"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with 2-hour timeout"
    echo "  $0 -t 7200 -c operator_configs/e2e_trace_config.py"
    echo ""
    echo "  # Run with disabled timeout (not recommended)"
    echo "  $0 -t none -c operator_configs/e2e_trace_config.py"
    echo ""
    echo "  # Run on specific GPUs with 4-hour timeout"
    echo "  $0 -t 14400 -c operator_configs/e2e_trace_config.py -g 0,1,2,3"
    echo ""
    echo "Recommended timeout values:"
    echo "  - Short sequences (<200 tokens):  1800s (30 min)"
    echo "  - Medium sequences (200-500):     3600s (1 hour)"
    echo "  - Long sequences (500-1000):      7200s (2 hours)"
    echo "  - Very long sequences (>1000):   14400s (4 hours)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -p|--phase)
            PHASE="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file is required${NC}"
    show_help
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Set GPU environment variable if specified
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    echo -e "${GREEN}Using GPUs: $GPUS${NC}"
fi

# Set timeout environment variable
if [ -n "$TIMEOUT" ]; then
    export VLLM_BENCHMARK_TIMEOUT="$TIMEOUT"
    if [ "$TIMEOUT" = "none" ] || [ "$TIMEOUT" = "0" ]; then
        echo -e "${YELLOW}⚠️  WARNING: Timeout disabled! Process may hang indefinitely.${NC}"
        echo -e "${YELLOW}    Make sure you can manually kill the process if needed.${NC}"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        TIMEOUT_MIN=$(($TIMEOUT / 60))
        echo -e "${GREEN}Timeout set to: ${TIMEOUT}s (${TIMEOUT_MIN} minutes)${NC}"
    fi
else
    echo -e "${GREEN}Using default adaptive timeout (15-109 minutes based on sequence length)${NC}"
fi

# Determine script to run based on phase
if [ "$PHASE" = "1" ]; then
    SCRIPT="benchmarks/benchmark_e2e_metrics.py"
elif [ "$PHASE" = "2" ]; then
    SCRIPT="benchmarks/benchmark_operators.py"
else
    echo -e "${RED}Error: Invalid phase $PHASE (must be 1 or 2)${NC}"
    exit 1
fi

# Show configuration summary
echo ""
echo "=========================================="
echo "Benchmark Configuration"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Phase: $PHASE"
echo "Script: $SCRIPT"
if [ -n "$GPUS" ]; then
    echo "GPUs: $GPUS"
else
    echo "GPUs: All available"
fi
if [ -n "$TIMEOUT" ]; then
    echo "Timeout: $TIMEOUT"
else
    echo "Timeout: Adaptive (default)"
fi
echo "=========================================="
echo ""

# Run benchmark
echo -e "${GREEN}Starting benchmark...${NC}"
echo ""

python "$SCRIPT" --config "$CONFIG_FILE" --phase "$PHASE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Benchmark completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Benchmark failed. Check logs above for details.${NC}"
    exit 1
fi




