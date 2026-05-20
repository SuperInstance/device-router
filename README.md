# device-router

> Heterogeneous compute router — auto-detect CUDA, iGPU, CPU, NPU and route ML workloads optimally.

Modern laptops and workstations have **multiple compute units**: a discrete GPU (CUDA), an integrated GPU (iGPU/DirectML), a Neural Processing Unit (NPU), and the CPU. Most ML frameworks pick one device and stick with it. **That's wasteful.**

`device-router` detects what's available and routes each workload to the best device automatically.

## Why it matters

| Workload | Best device | Why |
|---|---|---|
| Single embedding | CPU | No GPU transfer overhead (~9μs) |
| Small model (int8) | CPU (VNNI) | CPU has dedicated VNNI instructions |
| Medium model batched | iGPU | Good compute, low power |
| Large model training | CUDA GPU | Parallelism + AMP |
| ONNX inference | CPU | ONNX Runtime is CPU-optimized |

## Install

```bash
pip install device-router
```

Optional dependencies:

```bash
pip install device-router[cuda]      # CUDA GPU detection via torch
pip install device-router[directml]  # iGPU detection via torch-directml
pip install device-router[all]       # Everything
pip install device-router[dev]       # pytest + numpy for development
```

## Quick start

```python
from device_router import DeviceRouter, RoutingStrategy

router = DeviceRouter()
router.detect()  # Finds CUDA, DirectML, CPU features, NPU

# Route a workload
decision = router.route(
    model_size=1_000_000,  # parameters
    batch_size=32,
    precision="fp32",      # or "fp16", "bf16", "int8"
    strategy=RoutingStrategy.AUTO,
)
print(f"Use {decision.device} ({decision.reason})")
# → Use cuda (Medium/large model (1,000,000 params) — GPU recommended)

# System overview
overview = router.overview()
# Returns: {cuda: {...}, cpu: {...}, igpu: {...}, npu: {...}}
```

## Routing strategies

| Strategy | Description | Use case |
|---|---|---|
| `AUTO` | Best guess based on model size & batch | Default |
| `LATENCY` | Optimize for single-sample speed | Real-time inference |
| `THROUGHPUT` | Optimize for batch processing | Batch jobs |
| `POWER` | Prefer CPU/iGPU for efficiency | Laptops, mobile |

## How it works

### Without any dependencies
`device-router` runs pure CPU detection:
- CPU architecture, core count, frequency
- Instruction set features (AVX, AVX2, AVX-512, VNNI, AMX, NEON, SSE4)
- This is enough to route small models optimally

### With `torch` installed
Adds CUDA detection:
- GPU count, name, VRAM, compute capability
- CUDA/cuDNN version
- Enables AMP and GPU benchmarking

### With `torch-directml` installed
Adds iGPU detection:
- DirectML device availability
- Enables iGPU offloading for medium workloads

### Routing decision logic

```
ONNX model → CPU (always)
Training → CUDA (if available) or CPU
Small model (<100K params) → CPU
  + int8 + VNNI → CPU with VNNI optimization
Medium model (100K-10M) → CUDA > DirectML > CPU
Large model (>10M) + batched → CUDA with AMP
```

## API

### `DeviceRouter`

```python
router = DeviceRouter()
router.detect()                    # Scan for devices
router.overview()                   # Get system overview
router.route(model_size, batch_size, precision, strategy)  # Route workload
router.assign("cuda")               # Get torch.device for device string
```

### `RoutingDecision`

```python
decision.device       # "cuda", "cpu", "directml", "npu"
decision.reason       # Human-readable explanation
decision.precision    # Recommended precision
decision.use_amp      # Whether to use mixed precision
decision.confidence   # Confidence (0-1)
```

## SuperInstance Mesh integration

```python
# entry_point: superinstance.plugins
def register_device_router(registry):
    from device_router import DeviceRouter
    registry.register("devices", "router", DeviceRouter)
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT


## Ecosystem

Part of the [SuperInstance](https://github.com/SuperInstance) ecosystem:

| Package | Description |
|---------|-------------|
| [plato-core](https://github.com/SuperInstance/plato-core) | Base types + mesh registry |
| [tensor-spline](https://github.com/SuperInstance/tensor-spline) | SplineLinear neural compression |
| [eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed) | 5-layer matching cascade |
| [plato-training](https://github.com/SuperInstance/plato-training) | Training monolith |
| [device-router](https://github.com/SuperInstance/device-router) | Heterogeneous compute routing |
| [triplet-miner](https://github.com/SuperInstance/triplet-miner) | Git-powered contrastive data |
| [micro-onnx](https://github.com/SuperInstance/micro-onnx) | ONNX export + benchmark |
