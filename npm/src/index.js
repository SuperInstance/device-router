/**
 * DeviceRouter — heterogeneous compute routing for ML workloads.
 * JavaScript/TypeScript version of the Python device-router package.
 */

const RoutingStrategy = {
  AUTO: 'auto',
  LATENCY: 'latency',
  THROUGHPUT: 'throughput',
  POWER: 'power',
};

const SMALL_MODEL_PARAMS = 100_000;
const LARGE_MODEL_PARAMS = 10_000_000;

class RoutingDecision {
  constructor({ device, deviceIndex = 0, reason = '', precision = 'fp32', strategy = RoutingStrategy.AUTO, useAmp = false, confidence = 1.0 }) {
    this.device = device;
    this.deviceIndex = deviceIndex;
    this.reason = reason;
    this.precision = precision;
    this.strategy = strategy;
    this.useAmp = useAmp;
    this.confidence = confidence;
  }
}

class DeviceRouter {
  constructor() {
    this._cuda = { available: false, devices: [] };
    this._cpu = { available: true, arch: null, cores: null, features: {} };
    this._directml = { available: false };
    this._npu = { available: false };
    this._detected = false;
  }

  detect() {
    this._cpu = {
      available: true,
      arch: typeof navigator !== 'undefined' ? navigator.platform : process?.arch || 'unknown',
      cores: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : require('os').cpus()?.length || null,
      features: {},
    };
    // WebGPU detection (browser)
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      this._cuda = { available: true, name: 'WebGPU' };
    }
    // Node.js: could detect via child_process calling nvidia-smi
    this._detected = true;
    return this.overview();
  }

  overview() {
    if (!this._detected) this.detect();
    return {
      cuda: this._cuda,
      cpu: this._cpu,
      igpu: { available: this._directml.available },
      npu: this._npu,
    };
  }

  route({ modelSize = 0, batchSize = 1, precision = 'fp32', strategy = RoutingStrategy.AUTO, isTraining = false, isOnnx = false } = {}) {
    if (!this._detected) this.detect();

    if (isOnnx) {
      return new RoutingDecision({ device: 'cpu', reason: 'ONNX models are optimized for CPU execution', precision, strategy });
    }

    if (isTraining) {
      if (this._cuda.available) {
        return new RoutingDecision({ device: 'cuda', reason: 'Training workloads benefit from GPU parallelism', precision: precision === 'fp32' ? 'bf16' : precision, strategy, useAmp: true });
      }
      return new RoutingDecision({ device: 'cpu', reason: 'No CUDA GPU available — training on CPU', precision, strategy, confidence: 0.5 });
    }

    if (modelSize < SMALL_MODEL_PARAMS) {
      return new RoutingDecision({ device: 'cpu', reason: `Small model (${modelSize.toLocaleString()} params) — CPU is sufficient`, precision, strategy });
    }

    if (this._cuda.available) {
      return new RoutingDecision({ device: 'cuda', reason: `Medium/large model (${modelSize.toLocaleString()} params) — GPU recommended`, precision: precision === 'fp32' ? 'fp16' : precision, strategy, useAmp: true });
    }

    return new RoutingDecision({ device: 'cpu', reason: 'No accelerator available — using CPU', precision, strategy, confidence: 0.6 });
  }
}

module.exports = { DeviceRouter, RoutingStrategy, RoutingDecision };
