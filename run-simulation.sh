#!/bin/bash
# Kubernetes Scheduler Simulator runner script for cluster-trace-gpu-v2023

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/clusterdata/cluster-trace-gpu-v2023"
TRACE_NAME="${1:-openb_pod_list_default}"
OUTPUT_DIR="${SCRIPT_DIR}/simulation-results/${TRACE_NAME}"
TRACE_DIR="${DATA_DIR}/${TRACE_NAME}"

echo "========================================"
echo "Kubernetes Scheduler Simulator Runner"
echo "========================================"
echo "Trace: $TRACE_NAME"
echo "Trace Path: $TRACE_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================"

# Verify trace exists
if [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: Trace directory not found: $TRACE_DIR"
    echo ""
    echo "Available traces:"
    ls -d "${DATA_DIR}"/openb_pod_list_* 2>/dev/null | xargs -I {} basename {} | sort
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary config files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

cat > "$TEMP_DIR/cluster-config.yaml" <<EOF
apiVersion: simon/v1alpha1
kind: Config
metadata:
  name: simon-gpu-cluster-config
spec:
  cluster:
    customConfig: /trace
  customConfig:
    shufflePod: false
    workloadTuningConfig:
      ratio: 0.9
      seed: 233
    typicalPodsConfig:
      isInvolvedCpuPods: true
      podPopularityThreshold: 95
      isConsideredGpuResWeight: false
EOF

cat > "$TEMP_DIR/scheduler-config.yaml" <<EOF
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
percentageOfNodesToScore: 100
profiles:
  - schedulerName: simon-scheduler
    plugins:
      filter:
        enabled:
          - name: Open-Gpu-Share
      score:
        disabled:
          - name: RandomScore
          - name: DotProductScore
          - name: GpuClusteringScore
          - name: GpuPackingScore
          - name: BestFitScore
          - name: FGDScore
          - name: ImageLocality
          - name: NodeAffinity
          - name: PodTopologySpread
          - name: TaintToleration
          - name: NodeResourcesBalancedAllocation
          - name: InterPodAffinity
          - name: NodeResourcesLeastAllocated
          - name: NodePreferAvoidPods
        enabled:
          - name: FGDScore
            weight: 1000
      reserve:
        enabled:
          - name: Open-Gpu-Share
      bind:
        disabled:
          - name: DefaultBinder
        enabled:
          - name: Simon
    pluginConfig:
      - name: FGDScore
        args:
          dimExtMethod: share
          normMethod: max
      - name: Open-Gpu-Share
        args:
          dimExtMethod: share
          normMethod: max
          gpuSelMethod: FGDScore
EOF

echo ""
echo "Running simulator with Docker..."
echo ""

# Run simulator in Docker
docker run --rm \
  -v "$TRACE_DIR:/trace" \
  -v "$TEMP_DIR:/config" \
  -v "$OUTPUT_DIR:/output" \
  -w /root/kubernetes-scheduler-simulator \
  qzweng/kubernetes-scheduler-simulator:atc23 \
  /root/kubernetes-scheduler-simulator/bin/simon apply \
    --extended-resources "gpu" \
    -f "/config/cluster-config.yaml" \
    -s "/config/scheduler-config.yaml"

echo ""
echo "========================================"
echo "Simulation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
