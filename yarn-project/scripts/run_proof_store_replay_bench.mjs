#!/usr/bin/env node
import { BBNativeRollupProver } from '@aztec/bb-prover';
import { createLogger } from '@aztec/foundation/log';
import { createProofStore } from '@aztec/prover-client/broker';
import { ProvingRequestType } from '@aztec/stdlib/proofs';

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

const logger = createLogger('gpu-proof-store-replay-bench');
const PROOF_PROFILE_FILENAME = 'proof-profile.json';
const PROOF_VERIFY_PROFILE_FILENAME = 'proof-verify-profile.json';

process.env.BB_PROOF_BENCH ??= '1';

const PROOF_TYPE_ORDER = [
  'PUBLIC_VM',
  'PUBLIC_CHONK_VERIFIER',
  'PARITY_BASE',
  'PARITY_ROOT',
  'PRIVATE_TX_BASE_ROLLUP',
  'PUBLIC_TX_BASE_ROLLUP',
  'TX_MERGE_ROLLUP',
  'BLOCK_ROOT_FIRST_ROLLUP',
  'BLOCK_ROOT_SINGLE_TX_FIRST_ROLLUP',
  'BLOCK_ROOT_EMPTY_TX_FIRST_ROLLUP',
  'BLOCK_ROOT_ROLLUP',
  'BLOCK_ROOT_SINGLE_TX_ROLLUP',
  'BLOCK_MERGE_ROLLUP',
  'CHECKPOINT_ROOT_ROLLUP',
  'CHECKPOINT_ROOT_SINGLE_BLOCK_ROLLUP',
  'CHECKPOINT_PADDING_ROLLUP',
  'CHECKPOINT_MERGE_ROLLUP',
  'ROOT_ROLLUP',
];

const GPU_ENV_KEYS = [
  'BB_GPU_MSM_PRECOMPUTE_FACTOR',
  'BB_GPU_MSM_MAX_BATCH_SIZE',
  'BB_GPU_MSM_PREWARM_SRS_POINTS',
  'BB_PROOF_BENCH_DEFER_PROFILE_WRITE',
  'BB_PROOF_BENCH_PERSISTENT_BB',
  'BB_BINARY_PATH',
  'ACVM_BINARY_PATH',
  'LD_LIBRARY_PATH',
  'CUDA_VISIBLE_DEVICES',
];

const GPU_SRS_PREWARM_POINTS_BY_PROOF_TYPE = {
  PUBLIC_CHONK_VERIFIER: 1 << 22,
  PARITY_BASE: 1 << 22,
  PARITY_ROOT: 1 << 22,
  PRIVATE_TX_BASE_ROLLUP: 1 << 22,
  BLOCK_ROOT_SINGLE_TX_FIRST_ROLLUP: 1 << 21,
  CHECKPOINT_ROOT_SINGLE_BLOCK_ROLLUP: 1 << 23,
  ROOT_ROLLUP: 1 << 24,
};

const PROOF_TYPE_ARTIFACTS = {
  PUBLIC_CHONK_VERIFIER: 'PublicChonkVerifier',
  PARITY_BASE: 'ParityBaseArtifact',
  PARITY_ROOT: 'ParityRootArtifact',
  PRIVATE_TX_BASE_ROLLUP: 'PrivateTxBaseRollupArtifact',
  PUBLIC_TX_BASE_ROLLUP: 'PublicTxBaseRollupArtifact',
  TX_MERGE_ROLLUP: 'TxMergeRollupArtifact',
  BLOCK_ROOT_FIRST_ROLLUP: 'BlockRootFirstRollupArtifact',
  BLOCK_ROOT_SINGLE_TX_FIRST_ROLLUP: 'BlockRootSingleTxFirstRollupArtifact',
  BLOCK_ROOT_EMPTY_TX_FIRST_ROLLUP: 'BlockRootEmptyTxFirstRollupArtifact',
  BLOCK_ROOT_ROLLUP: 'BlockRootRollupArtifact',
  BLOCK_ROOT_SINGLE_TX_ROLLUP: 'BlockRootSingleTxRollupArtifact',
  BLOCK_MERGE_ROLLUP: 'BlockMergeRollupArtifact',
  CHECKPOINT_ROOT_ROLLUP: 'CheckpointRootRollupArtifact',
  CHECKPOINT_ROOT_SINGLE_BLOCK_ROLLUP: 'CheckpointRootSingleBlockRollupArtifact',
  CHECKPOINT_PADDING_ROLLUP: 'CheckpointPaddingRollupArtifact',
  CHECKPOINT_MERGE_ROLLUP: 'CheckpointMergeRollupArtifact',
  ROOT_ROLLUP: 'RootRollupArtifact',
};

function parseArgs() {
  const args = {
    repeats: 1,
    warmups: 0,
    includeTypes: undefined,
    excludeTypes: new Set(),
    gpuSrsPrewarmByType: false,
    persistentBbWorker: false,
    list: false,
  };
  for (let i = 2; i < process.argv.length; i++) {
    const key = process.argv[i];
    const value = process.argv[i + 1];
    switch (key) {
      case '--proof-store':
        args.proofStore = value;
        i++;
        break;
      case '--bb-bin':
        args.bbBin = value;
        i++;
        break;
      case '--acvm-bin':
        args.acvmBin = value;
        i++;
        break;
      case '--output-dir':
        args.outputDir = value;
        i++;
        break;
      case '--repeats':
        args.repeats = Number(value);
        i++;
        break;
      case '--warmups':
        args.warmups = Number(value);
        i++;
        break;
      case '--include-types':
        args.includeTypes = new Set(parseTypeList(value));
        i++;
        break;
      case '--exclude-types':
        args.excludeTypes = new Set(parseTypeList(value));
        i++;
        break;
      case '--gpu-srs-prewarm-by-type':
        args.gpuSrsPrewarmByType = true;
        break;
      case '--persistent-bb-worker':
        args.persistentBbWorker = true;
        break;
      case '--list':
        args.list = true;
        break;
      default:
        throw new Error(`Unknown argument: ${key}`);
    }
  }

  for (const key of ['proofStore', 'outputDir']) {
    if (!args[key]) {
      throw new Error(`Missing required argument --${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`);
    }
  }
  if (!args.list) {
    for (const key of ['bbBin', 'acvmBin']) {
      if (!args[key]) {
        throw new Error(`Missing required argument --${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`);
      }
    }
  }
  return args;
}

function maxGpuSrsPrewarmPointsForJobs(jobs) {
  return jobs.reduce((max, job) => Math.max(max, GPU_SRS_PREWARM_POINTS_BY_PROOF_TYPE[job.typeName] ?? 0), 0);
}

function parseTypeList(value) {
  return value
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
    .map(typeName => {
      if (ProvingRequestType[typeName] === undefined) {
        throw new Error(`Unknown proof type: ${typeName}`);
      }
      return typeName;
    });
}

function proofStoreRootPath(proofStoreUri) {
  const url = new URL(proofStoreUri);
  if (url.protocol !== 'file:') {
    throw new Error(`Proof-store discovery currently requires a file:// URI, got ${proofStoreUri}`);
  }
  if (url.host) {
    throw new Error(`Local file proof store cannot include a host, got ${proofStoreUri}`);
  }
  return fileURLToPath(url);
}

function proofTypeName(type) {
  return ProvingRequestType[type] ?? `UNKNOWN_${type}`;
}

function proofTypeSortIndex(typeName) {
  const index = PROOF_TYPE_ORDER.indexOf(typeName);
  return index === -1 ? PROOF_TYPE_ORDER.length : index;
}

async function sha256File(path) {
  const data = await readFile(path);
  return createHash('sha256').update(data).digest('hex');
}

function sha256Buffer(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

async function maybeFileStats(path) {
  try {
    const data = await readFile(path);
    return {
      bytes: data.length,
      gzipBytes: gzipSync(data).length,
      sha256: sha256Buffer(data),
    };
  } catch {
    return {
      bytes: null,
      gzipBytes: null,
      sha256: null,
    };
  }
}

function extractProofBuffer(result) {
  const proof = result?.proof ?? result;
  const buffer = proof?.binaryProof?.buffer;
  return Buffer.isBuffer(buffer) ? buffer : undefined;
}

function emptyNativeProfileSummary() {
  return {
    nativeProofs: 0,
    witnessGenerationMs: 0,
    bbProveMs: 0,
    bbVerifyMs: 0,
    bbBenchStagesMs: {
      oinkProverMs: 0,
      wireCommitmentsMs: 0,
      sortedListAccumulatorMs: 0,
      logDerivativeInverseMs: 0,
      grandProductMs: 0,
      ultraHonkApiProveMs: 0,
      sumcheckMs: 0,
      pcsMs: 0,
      commitmentKeyMs: 0,
      gpuSrsUploadMs: 0,
      gpuMsmMemoryFitMs: 0,
      gpuMsmRawMs: 0,
      gpuMsmRawBatchMs: 0,
    },
    bbBenchTopOps: [],
  };
}

function benchCategoryForKey(key) {
  if (key === 'UltraHonkAPI::prove') {
    return 'ultraHonkApiProveMs';
  }
  if (key === 'GPU::srs_upload') {
    return 'gpuSrsUploadMs';
  }
  if (key === 'GPU::msm_memory_fit') {
    return 'gpuMsmMemoryFitMs';
  }
  if (key === 'GPU::msm_raw') {
    return 'gpuMsmRawMs';
  }
  if (key === 'GPU::msm_raw_batch') {
    return 'gpuMsmRawBatchMs';
  }
  if (key.includes('OinkProver::prove')) {
    return 'oinkProverMs';
  }
  if (key.includes('execute_wire_commitments_round')) {
    return 'wireCommitmentsMs';
  }
  if (key.includes('execute_sorted_list_accumulator_round')) {
    return 'sortedListAccumulatorMs';
  }
  if (key.includes('execute_log_derivative_inverse_round')) {
    return 'logDerivativeInverseMs';
  }
  if (key.includes('execute_grand_product_computation_round')) {
    return 'grandProductMs';
  }
  if (key.toLowerCase().includes('sumcheck')) {
    return 'sumcheckMs';
  }
  if (key.includes('CommitmentKey::')) {
    return 'commitmentKeyMs';
  }
  if (
    key.includes('Shplemini') ||
    key.includes('Gemini') ||
    key.includes('KZG') ||
    key.includes('IPA') ||
    key.includes('PCS') ||
    key === 'compute_batched'
  ) {
    return 'pcsMs';
  }
  return undefined;
}

function addNativeProfileSummaries(left, right) {
  const result = emptyNativeProfileSummary();
  result.nativeProofs = left.nativeProofs + right.nativeProofs;
  result.witnessGenerationMs = left.witnessGenerationMs + right.witnessGenerationMs;
  result.bbProveMs = left.bbProveMs + right.bbProveMs;
  result.bbVerifyMs = left.bbVerifyMs + right.bbVerifyMs;
  for (const key of Object.keys(result.bbBenchStagesMs)) {
    result.bbBenchStagesMs[key] = (left.bbBenchStagesMs[key] ?? 0) + (right.bbBenchStagesMs[key] ?? 0);
  }
  const topOps = new Map();
  for (const op of [...left.bbBenchTopOps, ...right.bbBenchTopOps]) {
    topOps.set(op.name, (topOps.get(op.name) ?? 0) + op.elapsedMs);
  }
  result.bbBenchTopOps = [...topOps.entries()]
    .map(([name, elapsedMs]) => ({ name, elapsedMs }))
    .sort((a, b) => b.elapsedMs - a.elapsedMs)
    .slice(0, 20);
  return result;
}

function summarizeNativeProfiles(nativeProfiles) {
  const summary = emptyNativeProfileSummary();
  const ops = new Map();
  summary.nativeProofs = nativeProfiles.length;

  for (const profile of nativeProfiles) {
    summary.witnessGenerationMs += profile.witnessGenerationMs ?? 0;
    summary.bbProveMs += profile.bbProveMs ?? 0;

    for (const [key, entries] of Object.entries(profile.bbBench ?? {})) {
      const elapsedMs = entries.reduce((sum, entry) => sum + (entry.time_max ?? entry.time ?? 0) / 1_000_000, 0);
      if (elapsedMs === 0) {
        continue;
      }
      ops.set(key, (ops.get(key) ?? 0) + elapsedMs);
      const category = benchCategoryForKey(key);
      if (category) {
        summary.bbBenchStagesMs[category] += elapsedMs;
      }
    }
  }

  summary.bbBenchTopOps = [...ops.entries()]
    .map(([name, elapsedMs]) => ({ name, elapsedMs }))
    .sort((a, b) => b.elapsedMs - a.elapsedMs)
    .slice(0, 20);
  return summary;
}

function summarizeVerifyProfiles(verifyProfiles) {
  return {
    bbVerifyMs: verifyProfiles.reduce((sum, profile) => sum + (profile.bbVerifyMs ?? 0), 0),
  };
}

async function listProfilePaths(rootPath, profileFilename) {
  const paths = [];
  async function visit(dir) {
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const entryPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        await visit(entryPath);
      } else if (entry.isFile() && entry.name === profileFilename) {
        paths.push(entryPath);
      }
    }
  }
  await visit(rootPath);
  return paths;
}

async function listProofProfilePaths(rootPath) {
  return await listProfilePaths(rootPath, PROOF_PROFILE_FILENAME);
}

async function listProofVerifyProfilePaths(rootPath) {
  return await listProfilePaths(rootPath, PROOF_VERIFY_PROFILE_FILENAME);
}

async function readNewProofProfiles(rootPath, existingPaths) {
  const paths = await listProofProfilePaths(rootPath);
  const profiles = [];
  for (const profilePath of paths) {
    if (existingPaths.has(profilePath)) {
      continue;
    }
    const profile = JSON.parse(await readFile(profilePath, 'utf8'));
    profiles.push({
      profilePath,
      circuitType: profile.circuitType,
      circuitName: profile.circuitName,
      witnessGenerationMs: profile.witnessGenerationMs ?? 0,
      bbProveMs: profile.bbProveMs ?? 0,
      bbBenchPath: profile.bbBenchPath,
      bbBench: profile.bbBench,
    });
  }
  return profiles;
}

async function readNewProofVerifyProfiles(rootPath, existingPaths) {
  const paths = await listProofVerifyProfilePaths(rootPath);
  const profiles = [];
  for (const profilePath of paths) {
    if (existingPaths.has(profilePath)) {
      continue;
    }
    const profile = JSON.parse(await readFile(profilePath, 'utf8'));
    profiles.push({
      profilePath,
      circuitType: profile.circuitType,
      circuitName: profile.circuitName,
      bbVerifyMs: profile.bbVerifyMs ?? 0,
    });
  }
  return profiles;
}

async function discoverProofInputs(args) {
  const rootPath = proofStoreRootPath(args.proofStore);
  const inputsPath = join(rootPath, 'inputs');
  const typeDirs = await readdir(inputsPath, { withFileTypes: true });
  const jobs = [];

  for (const typeDir of typeDirs) {
    if (!typeDir.isDirectory()) {
      continue;
    }
    const typeName = typeDir.name;
    if (ProvingRequestType[typeName] === undefined) {
      throw new Error(`Unknown proof input type directory: ${typeName}`);
    }
    if (args.includeTypes && !args.includeTypes.has(typeName)) {
      continue;
    }
    if (args.excludeTypes.has(typeName)) {
      continue;
    }

    const typePath = join(inputsPath, typeName);
    const entries = await readdir(typePath, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) {
        continue;
      }
      const inputPath = join(typePath, entry.name);
      const inputStat = await stat(inputPath);
      jobs.push({
        type: ProvingRequestType[typeName],
        typeName,
        jobId: entry.name,
        inputPath,
        proofUri: pathToFileURL(inputPath).href,
        inputSizeBytes: inputStat.size,
        inputSha256: await sha256File(inputPath),
      });
    }
  }

  jobs.sort((a, b) => {
    const typeDelta = proofTypeSortIndex(a.typeName) - proofTypeSortIndex(b.typeName);
    if (typeDelta !== 0) {
      return typeDelta;
    }
    return a.proofUri.localeCompare(b.proofUri);
  });
  return jobs;
}

async function dispatchProof(prover, type, inputs) {
  switch (type) {
    case ProvingRequestType.PUBLIC_VM:
      return await prover.getAvmProof(inputs);
    case ProvingRequestType.PUBLIC_CHONK_VERIFIER:
      return await prover.getPublicChonkVerifierProof(inputs);
    case ProvingRequestType.PRIVATE_TX_BASE_ROLLUP:
      return await prover.getPrivateTxBaseRollupProof(inputs);
    case ProvingRequestType.PUBLIC_TX_BASE_ROLLUP:
      return await prover.getPublicTxBaseRollupProof(inputs);
    case ProvingRequestType.TX_MERGE_ROLLUP:
      return await prover.getTxMergeRollupProof(inputs);
    case ProvingRequestType.BLOCK_ROOT_FIRST_ROLLUP:
      return await prover.getBlockRootFirstRollupProof(inputs);
    case ProvingRequestType.BLOCK_ROOT_SINGLE_TX_FIRST_ROLLUP:
      return await prover.getBlockRootSingleTxFirstRollupProof(inputs);
    case ProvingRequestType.BLOCK_ROOT_EMPTY_TX_FIRST_ROLLUP:
      return await prover.getBlockRootEmptyTxFirstRollupProof(inputs);
    case ProvingRequestType.BLOCK_ROOT_ROLLUP:
      return await prover.getBlockRootRollupProof(inputs);
    case ProvingRequestType.BLOCK_ROOT_SINGLE_TX_ROLLUP:
      return await prover.getBlockRootSingleTxRollupProof(inputs);
    case ProvingRequestType.BLOCK_MERGE_ROLLUP:
      return await prover.getBlockMergeRollupProof(inputs);
    case ProvingRequestType.CHECKPOINT_ROOT_ROLLUP:
      return await prover.getCheckpointRootRollupProof(inputs);
    case ProvingRequestType.CHECKPOINT_ROOT_SINGLE_BLOCK_ROLLUP:
      return await prover.getCheckpointRootSingleBlockRollupProof(inputs);
    case ProvingRequestType.CHECKPOINT_PADDING_ROLLUP:
      return await prover.getCheckpointPaddingRollupProof(inputs);
    case ProvingRequestType.CHECKPOINT_MERGE_ROLLUP:
      return await prover.getCheckpointMergeRollupProof(inputs);
    case ProvingRequestType.ROOT_ROLLUP:
      return await prover.getRootRollupProof(inputs);
    case ProvingRequestType.PARITY_BASE:
      return await prover.getBaseParityProof(inputs);
    case ProvingRequestType.PARITY_ROOT:
      return await prover.getRootParityProof(inputs);
    default:
      throw new Error(`Unsupported proof type for this benchmark: ${proofTypeName(type)}`);
  }
}

async function prewarmCircuitAssets(prover, jobs) {
  const artifacts = new Set();
  for (const job of jobs) {
    const artifact = PROOF_TYPE_ARTIFACTS[job.typeName];
    if (artifact) {
      artifacts.add(artifact);
    }
  }

  let prewarmMs = 0;
  for (const artifact of artifacts) {
    prewarmMs += await prover.prewarmProofCircuit(artifact);
  }
  return prewarmMs;
}

async function runSuite(args, proofStore, jobs, repeat, warmup, rawPath) {
  const runName = warmup ? `warmup_${repeat.toString().padStart(2, '0')}` : `run_${repeat.toString().padStart(2, '0')}`;
  const runDir = resolve(args.outputDir, runName);
  const bbDir = join(runDir, 'bb');
  const acvmDir = join(runDir, 'acvm');

  const setupStart = performance.now();
  await mkdir(bbDir, { recursive: true });
  await mkdir(acvmDir, { recursive: true });

  const prover = await BBNativeRollupProver.new({
    bbBinaryPath: args.bbBin,
    bbWorkingDirectory: bbDir,
    acvmBinaryPath: args.acvmBin,
    acvmWorkingDirectory: acvmDir,
    bbSkipCleanup: true,
  });
  const gpuSrsPrewarmSetupStart = performance.now();
  const gpuSrsPrewarmPoints = args.persistentBbWorker ? maxGpuSrsPrewarmPointsForJobs(jobs) : 0;
  const gpuSrsPrewarmSetupMs = gpuSrsPrewarmPoints > 0 ? await prover.prewarmGpuSrs(gpuSrsPrewarmPoints) : 0;
  const gpuSrsPrewarmSetupWallMs = performance.now() - gpuSrsPrewarmSetupStart;
  const circuitAssetPrewarmSetupStart = performance.now();
  const circuitAssetPrewarmSetupMs = args.persistentBbWorker ? await prewarmCircuitAssets(prover, jobs) : 0;
  const circuitAssetPrewarmSetupWallMs = performance.now() - circuitAssetPrewarmSetupStart;
  const setupMs = performance.now() - setupStart;

  const suiteStart = performance.now();
  const records = [];
  for (let jobIndex = 0; jobIndex < jobs.length; jobIndex++) {
    const jobRef = jobs[jobIndex];
    const knownProofProfiles = new Set(await listProofProfilePaths(bbDir));
    const knownProofVerifyProfiles = new Set(await listProofVerifyProfilePaths(bbDir));
    const jobStart = performance.now();
    const inputLoadStart = performance.now();
    const job = await proofStore.getProofInput(jobRef.proofUri);
    const inputLoadMs = performance.now() - inputLoadStart;
    if (job.type !== jobRef.type) {
      throw new Error(
        `Discovered ${jobRef.typeName} for ${jobRef.proofUri}, but proof store decoded ${proofTypeName(job.type)}`,
      );
    }

    if (args.gpuSrsPrewarmByType && !args.persistentBbWorker) {
      const prewarmPoints = GPU_SRS_PREWARM_POINTS_BY_PROOF_TYPE[jobRef.typeName];
      if (prewarmPoints === undefined) {
        delete process.env.BB_GPU_MSM_PREWARM_SRS_POINTS;
      } else {
        process.env.BB_GPU_MSM_PREWARM_SRS_POINTS = String(prewarmPoints);
      }
    }

    const proofGenerationStart = performance.now();
    const result = await dispatchProof(prover, job.type, job.inputs);
    const proofGenerationMs = performance.now() - proofGenerationStart;

    const proofOutputStart = performance.now();
    const proofBuffer = extractProofBuffer(result);
    const proofStats = await maybeFileStats(join(bbDir, 'proof'));
    const proofOutputMs = performance.now() - proofOutputStart;
    const elapsedMs = performance.now() - jobStart;
    const proofProfileWriteStart = performance.now();
    await prover.flushDeferredProofProfiles();
    const proofProfileWriteMs = performance.now() - proofProfileWriteStart;
    const proofProfileReadStart = performance.now();
    const nativeProofProfiles = await readNewProofProfiles(bbDir, knownProofProfiles);
    const proofVerifyProfiles = await readNewProofVerifyProfiles(bbDir, knownProofVerifyProfiles);
    const nativeProfileSummary = summarizeNativeProfiles(nativeProofProfiles);
    const verifyProfileSummary = summarizeVerifyProfiles(proofVerifyProfiles);
    nativeProfileSummary.bbVerifyMs = verifyProfileSummary.bbVerifyMs;
    const proofProfileReadMs = performance.now() - proofProfileReadStart;
    const proofInternalOverheadMs =
      proofGenerationMs -
      nativeProfileSummary.witnessGenerationMs -
      nativeProfileSummary.bbProveMs -
      nativeProfileSummary.bbVerifyMs;
    const stageTotalMs = inputLoadMs + proofGenerationMs + proofOutputMs;
    const record = {
      benchmark: 'proof-store-replay-job',
      repeat,
      warmup,
      jobIndex,
      proofType: jobRef.typeName,
      jobId: jobRef.jobId,
      proofUri: jobRef.proofUri,
      inputPath: jobRef.inputPath,
      inputSizeBytes: jobRef.inputSizeBytes,
      inputSha256: jobRef.inputSha256,
      elapsedMs,
      inputLoadMs,
      proofGenerationMs,
      proofProfileWriteMs,
      proofProfileReadMs,
      nativeProofs: nativeProfileSummary.nativeProofs,
      witnessGenerationMs: nativeProfileSummary.witnessGenerationMs,
      bbProveMs: nativeProfileSummary.bbProveMs,
      bbVerifyMs: nativeProfileSummary.bbVerifyMs,
      proofInternalOverheadMs,
      bbBenchStagesMs: nativeProfileSummary.bbBenchStagesMs,
      bbBenchTopOps: nativeProfileSummary.bbBenchTopOps,
      nativeProofProfiles: nativeProofProfiles.map(profile => ({
        profilePath: profile.profilePath,
        circuitType: profile.circuitType,
        circuitName: profile.circuitName,
        witnessGenerationMs: profile.witnessGenerationMs,
        bbProveMs: profile.bbProveMs,
        bbBenchPath: profile.bbBenchPath,
      })),
      proofVerifyProfiles,
      proofOutputMs,
      stageTotalMs,
      overheadMs: elapsedMs - stageTotalMs,
      verified: true,
      proofSizeBytes: proofBuffer?.length ?? proofStats.bytes,
      proofGzipSizeBytes: proofBuffer ? gzipSync(proofBuffer).length : proofStats.gzipBytes,
      proofSha256: proofBuffer ? sha256Buffer(proofBuffer) : proofStats.sha256,
      runDir,
      bbDir,
      acvmDir,
      bbBin: args.bbBin,
      acvmBin: args.acvmBin,
      persistentBbWorker: args.persistentBbWorker,
    };
    records.push(record);
    await writeFile(rawPath, JSON.stringify(record) + '\n', { flag: 'a' });
    console.log(
      `${warmup ? 'warmup' : 'run'} ${repeat} job ${jobIndex + 1}/${jobs.length} ${jobRef.typeName} ${elapsedMs.toFixed(
        3,
      )} ms witgen=${nativeProfileSummary.witnessGenerationMs.toFixed(3)} ms bb=${nativeProfileSummary.bbProveMs.toFixed(
        3,
      )} ms api=${nativeProfileSummary.bbBenchStagesMs.ultraHonkApiProveMs.toFixed(
        3,
      )} ms verify=${nativeProfileSummary.bbVerifyMs.toFixed(3)} ms proof-overhead=${proofInternalOverheadMs.toFixed(
        3,
      )} ms profile-write=${proofProfileWriteMs.toFixed(
        3,
      )} ms profile-read=${proofProfileReadMs.toFixed(
        3,
      )} ms overhead=${(elapsedMs - stageTotalMs).toFixed(3)} ms`,
    );
  }

  const jobLoopWallMs = performance.now() - suiteStart;
  const jobLoopElapsedMs = records.reduce((sum, record) => sum + record.elapsedMs, 0);
  const inputLoadMs = records.reduce((sum, record) => sum + record.inputLoadMs, 0);
  const proofGenerationMs = records.reduce((sum, record) => sum + record.proofGenerationMs, 0);
  const proofProfileWriteMs = records.reduce((sum, record) => sum + record.proofProfileWriteMs, 0);
  const proofProfileReadMs = records.reduce((sum, record) => sum + record.proofProfileReadMs, 0);
  const nativeProfileSummary = records
    .map(record => ({
      nativeProofs: record.nativeProofs,
      witnessGenerationMs: record.witnessGenerationMs,
      bbProveMs: record.bbProveMs,
      bbVerifyMs: record.bbVerifyMs,
      bbBenchStagesMs: record.bbBenchStagesMs,
      bbBenchTopOps: record.bbBenchTopOps,
    }))
    .reduce(addNativeProfileSummaries, emptyNativeProfileSummary());
  const proofInternalOverheadMs = records.reduce((sum, record) => sum + record.proofInternalOverheadMs, 0);
  const proofOutputMs = records.reduce((sum, record) => sum + record.proofOutputMs, 0);
  const stageTotalMs = inputLoadMs + proofGenerationMs + proofOutputMs;
  const elapsedMs = jobLoopElapsedMs;
  const elapsedWithSetupMs = setupMs + elapsedMs;
  const suiteRecord = {
    benchmark: 'proof-store-replay-suite',
    repeat,
    warmup,
    elapsedMs,
    elapsedWithSetupMs,
    setupMs,
    gpuSrsPrewarmSetupMs,
    gpuSrsPrewarmSetupWallMs,
    circuitAssetPrewarmSetupMs,
    circuitAssetPrewarmSetupWallMs,
    gpuSrsPrewarmPoints,
    jobLoopElapsedMs,
    jobLoopWallMs,
    inputLoadMs,
    proofGenerationMs,
    proofProfileWriteMs,
    proofProfileReadMs,
    nativeProofs: nativeProfileSummary.nativeProofs,
    witnessGenerationMs: nativeProfileSummary.witnessGenerationMs,
    bbProveMs: nativeProfileSummary.bbProveMs,
    bbVerifyMs: nativeProfileSummary.bbVerifyMs,
    proofInternalOverheadMs,
    bbBenchStagesMs: nativeProfileSummary.bbBenchStagesMs,
    bbBenchTopOps: nativeProfileSummary.bbBenchTopOps,
    proofOutputMs,
    stageTotalMs,
    overheadMs: elapsedMs - stageTotalMs,
    jobs: records.length,
    runDir,
    bbDir,
    acvmDir,
  };
  await writeFile(rawPath, JSON.stringify(suiteRecord) + '\n', { flag: 'a' });
  await prover.stop();
  return { suiteRecord, records };
}

function avg(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdev(values) {
  const mean = avg(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

function groupBy(records, keyFn) {
  const groups = new Map();
  for (const record of records) {
    const key = keyFn(record);
    const group = groups.get(key) ?? [];
    group.push(record);
    groups.set(key, group);
  }
  return groups;
}

function summarizeNativeRecordGroup(records) {
  return records
    .map(record => ({
      nativeProofs: record.nativeProofs ?? 0,
      witnessGenerationMs: record.witnessGenerationMs ?? 0,
      bbProveMs: record.bbProveMs ?? 0,
      bbVerifyMs: record.bbVerifyMs ?? 0,
      bbBenchStagesMs: record.bbBenchStagesMs ?? emptyNativeProfileSummary().bbBenchStagesMs,
      bbBenchTopOps: record.bbBenchTopOps ?? [],
    }))
    .reduce(addNativeProfileSummaries, emptyNativeProfileSummary());
}

function safeExec(command, args) {
  try {
    return execFileSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return null;
  }
}

function collectMetadata(args, jobs) {
  const env = Object.fromEntries(GPU_ENV_KEYS.map(key => [key, process.env[key] ?? null]));
  return {
    benchmark: 'proof-store-replay',
    command: process.argv,
    cwd: process.cwd(),
    node: process.version,
    proofStore: args.proofStore,
    bbBin: args.bbBin ?? null,
    acvmBin: args.acvmBin ?? null,
    outputDir: args.outputDir,
    repeats: args.repeats,
    warmups: args.warmups,
    includeTypes: args.includeTypes ? [...args.includeTypes] : null,
    excludeTypes: [...args.excludeTypes],
    gpuSrsPrewarmByType: args.gpuSrsPrewarmByType,
    persistentBbWorker: args.persistentBbWorker,
    env,
    git: {
      commit: safeExec('git', ['rev-parse', 'HEAD']),
      status: safeExec('git', ['status', '--short', '--branch']),
    },
    nvidiaSmi: safeExec('nvidia-smi', []),
    discoveredJobs: jobs.map(job => ({
      proofType: job.typeName,
      jobId: job.jobId,
      proofUri: job.proofUri,
      inputSizeBytes: job.inputSizeBytes,
      inputSha256: job.inputSha256,
    })),
  };
}

function summarize(args, jobs, suiteRecords, jobRecords) {
  const measuredSuites = suiteRecords.filter(record => !record.warmup);
  const measuredJobs = jobRecords.filter(record => !record.warmup);
  const suiteElapsed = measuredSuites.map(record => record.elapsedMs);
  const suiteElapsedWithSetup = measuredSuites.map(record => record.elapsedWithSetupMs ?? record.elapsedMs);
  const suiteSetup = measuredSuites.map(record => record.setupMs);
  const suiteAvgField = field => avg(measuredSuites.map(record => record[field]));
  const suiteNative = summarizeNativeRecordGroup(measuredSuites);
  const byType = [...groupBy(measuredJobs, record => record.proofType)].map(([proofType, records]) => {
    const elapsed = records.map(record => record.elapsedMs);
    const avgFor = field => avg(records.map(record => record[field] ?? 0));
    const avgStage = field => avg(records.map(record => record.bbBenchStagesMs?.[field] ?? 0));
    return {
      proofType,
      samples: records.length,
      jobsPerRepeat: records.length / measuredSuites.length,
      elapsedMsAvg: avg(elapsed),
      elapsedMsMin: Math.min(...elapsed),
      elapsedMsMax: Math.max(...elapsed),
      elapsedMsStdev: stdev(elapsed),
      inputLoadMsAvg: avgFor('inputLoadMs'),
      proofGenerationMsAvg: avgFor('proofGenerationMs'),
      proofProfileWriteMsAvg: avgFor('proofProfileWriteMs'),
      proofProfileReadMsAvg: avgFor('proofProfileReadMs'),
      proofOutputMsAvg: avgFor('proofOutputMs'),
      stageTotalMsAvg: avgFor('stageTotalMs'),
      overheadMsAvg: avgFor('overheadMs'),
      nativeProofsAvg: avgFor('nativeProofs'),
      witnessGenerationMsAvg: avgFor('witnessGenerationMs'),
      bbProveMsAvg: avgFor('bbProveMs'),
      bbVerifyMsAvg: avgFor('bbVerifyMs'),
      proofInternalOverheadMsAvg: avgFor('proofInternalOverheadMs'),
      ultraHonkApiProveMsAvg: avgStage('ultraHonkApiProveMs'),
      oinkProverMsAvg: avgStage('oinkProverMs'),
      wireCommitmentsMsAvg: avgStage('wireCommitmentsMs'),
      sortedListAccumulatorMsAvg: avgStage('sortedListAccumulatorMs'),
      logDerivativeInverseMsAvg: avgStage('logDerivativeInverseMs'),
      grandProductMsAvg: avgStage('grandProductMs'),
      sumcheckMsAvg: avgStage('sumcheckMs'),
      pcsMsAvg: avgStage('pcsMs'),
      commitmentKeyMsAvg: avgStage('commitmentKeyMs'),
      gpuSrsUploadMsAvg: avgStage('gpuSrsUploadMs'),
      gpuMsmMemoryFitMsAvg: avgStage('gpuMsmMemoryFitMs'),
      gpuMsmRawMsAvg: avgStage('gpuMsmRawMs'),
      gpuMsmRawBatchMsAvg: avgStage('gpuMsmRawBatchMs'),
    };
  });
  byType.sort((a, b) => proofTypeSortIndex(a.proofType) - proofTypeSortIndex(b.proofType));

  return {
    benchmark: 'proof-store-replay',
    outputDir: args.outputDir,
    proofStore: args.proofStore,
    samples: measuredSuites.length,
    jobsPerRepeat: jobs.length,
    elapsedMsAvg: avg(suiteElapsed),
    elapsedMsMin: Math.min(...suiteElapsed),
    elapsedMsMax: Math.max(...suiteElapsed),
    elapsedMsStdev: stdev(suiteElapsed),
    elapsedWithSetupMsAvg: avg(suiteElapsedWithSetup),
    setupMsAvg: avg(suiteSetup),
    gpuSrsPrewarmSetupMsAvg: suiteAvgField('gpuSrsPrewarmSetupMs'),
    gpuSrsPrewarmSetupWallMsAvg: suiteAvgField('gpuSrsPrewarmSetupWallMs'),
    circuitAssetPrewarmSetupMsAvg: suiteAvgField('circuitAssetPrewarmSetupMs'),
    circuitAssetPrewarmSetupWallMsAvg: suiteAvgField('circuitAssetPrewarmSetupWallMs'),
    gpuSrsPrewarmPointsMax: Math.max(...measuredSuites.map(record => record.gpuSrsPrewarmPoints ?? 0)),
    inputLoadMsAvg: suiteAvgField('inputLoadMs'),
    proofGenerationMsAvg: suiteAvgField('proofGenerationMs'),
    proofProfileWriteMsAvg: suiteAvgField('proofProfileWriteMs'),
    proofProfileReadMsAvg: suiteAvgField('proofProfileReadMs'),
    proofOutputMsAvg: suiteAvgField('proofOutputMs'),
    stageTotalMsAvg: suiteAvgField('stageTotalMs'),
    overheadMsAvg: suiteAvgField('overheadMs'),
    nativeProofsTotal: suiteNative.nativeProofs,
    witnessGenerationMsTotal: suiteNative.witnessGenerationMs,
    bbProveMsTotal: suiteNative.bbProveMs,
    bbVerifyMsTotal: suiteNative.bbVerifyMs,
    proofInternalOverheadMsTotal: measuredJobs.reduce((sum, record) => sum + (record.proofInternalOverheadMs ?? 0), 0),
    bbBenchStagesMsTotal: suiteNative.bbBenchStagesMs,
    bbBenchTopOps: suiteNative.bbBenchTopOps,
    suiteRecords: measuredSuites,
    byType,
    records: measuredJobs,
  };
}

function formatMs(value) {
  return (value ?? 0).toFixed(3);
}

async function writeSummaryMd(outputDir, summary) {
  const lines = [
    '# Proof Store Replay Benchmark',
    '',
    `Proof store: \`${summary.proofStore}\``,
    '',
    '| metric | value |',
    '|---|---:|',
    `| measured repeats | ${summary.samples} |`,
    `| jobs per repeat | ${summary.jobsPerRepeat} |`,
    `| total avg ms | ${formatMs(summary.elapsedMsAvg)} |`,
    `| total incl setup avg ms | ${formatMs(summary.elapsedWithSetupMsAvg)} |`,
    `| total min ms | ${formatMs(summary.elapsedMsMin)} |`,
    `| total max ms | ${formatMs(summary.elapsedMsMax)} |`,
    `| total stdev ms | ${formatMs(summary.elapsedMsStdev)} |`,
    `| setup avg ms | ${formatMs(summary.setupMsAvg)} |`,
    `| gpu srs prewarm setup avg ms | ${formatMs(summary.gpuSrsPrewarmSetupMsAvg)} |`,
    `| gpu srs prewarm setup wall avg ms | ${formatMs(summary.gpuSrsPrewarmSetupWallMsAvg)} |`,
    `| circuit asset prewarm setup avg ms | ${formatMs(summary.circuitAssetPrewarmSetupMsAvg)} |`,
    `| circuit asset prewarm setup wall avg ms | ${formatMs(summary.circuitAssetPrewarmSetupWallMsAvg)} |`,
    `| gpu srs prewarm points max | ${summary.gpuSrsPrewarmPointsMax} |`,
    `| input load avg ms | ${formatMs(summary.inputLoadMsAvg)} |`,
    `| proof generation avg ms | ${formatMs(summary.proofGenerationMsAvg)} |`,
    `| proof profile write avg ms | ${formatMs(summary.proofProfileWriteMsAvg)} |`,
    `| proof profile read avg ms | ${formatMs(summary.proofProfileReadMsAvg)} |`,
    `| native proofs total | ${summary.nativeProofsTotal} |`,
    `| witgen total ms | ${formatMs(summary.witnessGenerationMsTotal)} |`,
    `| bb prove total ms | ${formatMs(summary.bbProveMsTotal)} |`,
    `| bb verify total ms | ${formatMs(summary.bbVerifyMsTotal)} |`,
    `| proof internal overhead total ms | ${formatMs(summary.proofInternalOverheadMsTotal)} |`,
    `| ultra honk api prove total ms | ${formatMs(summary.bbBenchStagesMsTotal.ultraHonkApiProveMs)} |`,
    `| oink total ms | ${formatMs(summary.bbBenchStagesMsTotal.oinkProverMs)} |`,
    `| wire commitments total ms | ${formatMs(summary.bbBenchStagesMsTotal.wireCommitmentsMs)} |`,
    `| sorted list accumulator total ms | ${formatMs(summary.bbBenchStagesMsTotal.sortedListAccumulatorMs)} |`,
    `| log-derivative inverse total ms | ${formatMs(summary.bbBenchStagesMsTotal.logDerivativeInverseMs)} |`,
    `| grand product total ms | ${formatMs(summary.bbBenchStagesMsTotal.grandProductMs)} |`,
    `| sumcheck total ms | ${formatMs(summary.bbBenchStagesMsTotal.sumcheckMs)} |`,
    `| pcs total ms | ${formatMs(summary.bbBenchStagesMsTotal.pcsMs)} |`,
    `| commitment key total ms | ${formatMs(summary.bbBenchStagesMsTotal.commitmentKeyMs)} |`,
    `| gpu srs upload total ms | ${formatMs(summary.bbBenchStagesMsTotal.gpuSrsUploadMs)} |`,
    `| gpu msm memory fit total ms | ${formatMs(summary.bbBenchStagesMsTotal.gpuMsmMemoryFitMs)} |`,
    `| gpu msm raw total ms | ${formatMs(summary.bbBenchStagesMsTotal.gpuMsmRawMs)} |`,
    `| gpu msm raw batch total ms | ${formatMs(summary.bbBenchStagesMsTotal.gpuMsmRawBatchMs)} |`,
    `| output avg ms | ${formatMs(summary.proofOutputMsAvg)} |`,
    `| stage total avg ms | ${formatMs(summary.stageTotalMsAvg)} |`,
    `| overhead avg ms | ${formatMs(summary.overheadMsAvg)} |`,
    '',
    '## By Type',
    '',
    '| proof type | samples | jobs/repeat | avg ms | native proofs | witgen ms | bb prove ms | ultra honk api prove ms | bb verify ms | proof overhead ms | profile write ms | profile read ms | oink ms | wire comm ms | sorted acc ms | log-derivative ms | grand product ms | sumcheck ms | pcs ms | commitment key ms | gpu srs upload ms | gpu msm fit ms | gpu msm raw ms | gpu msm raw batch ms | input load ms | output ms | overhead ms | min ms | max ms | stdev ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ...summary.byType.map(
      row =>
        `| ${row.proofType} | ${row.samples} | ${row.jobsPerRepeat} | ${formatMs(row.elapsedMsAvg)} | ${formatMs(
          row.nativeProofsAvg,
        )} | ${formatMs(
          row.witnessGenerationMsAvg,
        )} | ${formatMs(row.bbProveMsAvg)} | ${formatMs(row.ultraHonkApiProveMsAvg)} | ${formatMs(
          row.bbVerifyMsAvg,
        )} | ${formatMs(
          row.proofInternalOverheadMsAvg,
        )} | ${formatMs(
          row.proofProfileWriteMsAvg,
        )} | ${formatMs(
          row.proofProfileReadMsAvg,
        )} | ${formatMs(row.oinkProverMsAvg)} | ${formatMs(
          row.wireCommitmentsMsAvg,
        )} | ${formatMs(row.sortedListAccumulatorMsAvg)} | ${formatMs(row.logDerivativeInverseMsAvg)} | ${formatMs(
          row.grandProductMsAvg,
        )} | ${formatMs(row.sumcheckMsAvg)} | ${formatMs(row.pcsMsAvg)} | ${formatMs(
          row.commitmentKeyMsAvg,
        )} | ${formatMs(row.gpuSrsUploadMsAvg)} | ${formatMs(row.gpuMsmMemoryFitMsAvg)} | ${formatMs(
          row.gpuMsmRawMsAvg,
        )} | ${formatMs(row.gpuMsmRawBatchMsAvg)} | ${formatMs(row.inputLoadMsAvg)} | ${formatMs(
          row.proofOutputMsAvg,
        )} | ${formatMs(row.overheadMsAvg)} | ${formatMs(row.elapsedMsMin)} | ${formatMs(
          row.elapsedMsMax,
        )} | ${formatMs(row.elapsedMsStdev)} |`,
    ),
    '',
    '## Top BB Ops',
    '',
    '| op | elapsed ms |',
    '|---|---:|',
    ...summary.bbBenchTopOps.map(op => `| ${op.name} | ${formatMs(op.elapsedMs)} |`),
    '',
    '## Jobs',
    '',
    '| repeat | index | proof type | elapsed ms | native proofs | witgen ms | bb prove ms | ultra honk api prove ms | bb verify ms | proof overhead ms | profile write ms | profile read ms | oink ms | sumcheck ms | pcs ms | commitment key ms | gpu srs upload ms | gpu msm fit ms | gpu msm raw ms | gpu msm raw batch ms | overhead ms | input load ms | output ms | input sha256 | proof sha256 | proof bytes |',
    '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|',
    ...summary.records.map(
      record =>
        `| ${record.repeat} | ${record.jobIndex} | ${record.proofType} | ${formatMs(record.elapsedMs)} | ${formatMs(
          record.nativeProofs,
        )} | ${formatMs(record.witnessGenerationMs)} | ${formatMs(record.bbProveMs)} | ${formatMs(
          record.bbBenchStagesMs?.ultraHonkApiProveMs,
        )} | ${formatMs(
          record.bbVerifyMs,
        )} | ${formatMs(record.proofInternalOverheadMs)} | ${formatMs(
          record.proofProfileWriteMs,
        )} | ${formatMs(
          record.proofProfileReadMs,
        )} | ${formatMs(
          record.bbBenchStagesMs?.oinkProverMs,
        )} | ${formatMs(record.bbBenchStagesMs?.sumcheckMs)} | ${formatMs(record.bbBenchStagesMs?.pcsMs)} | ${formatMs(
          record.bbBenchStagesMs?.commitmentKeyMs,
        )} | ${formatMs(record.bbBenchStagesMs?.gpuSrsUploadMs)} | ${formatMs(
          record.bbBenchStagesMs?.gpuMsmMemoryFitMs,
        )} | ${formatMs(record.bbBenchStagesMs?.gpuMsmRawMs)} | ${formatMs(
          record.bbBenchStagesMs?.gpuMsmRawBatchMs,
        )} | ${formatMs(
          record.overheadMs,
        )} | ${formatMs(record.inputLoadMs)} | ${formatMs(
          record.proofOutputMs,
        )} | ${record.inputSha256} | ${record.proofSha256 ?? ''} | ${record.proofSizeBytes ?? ''} |`,
    ),
    '',
  ];
  await writeFile(join(outputDir, 'summary.md'), lines.join('\n'));
}

async function main() {
  const args = parseArgs();
  args.outputDir = resolve(args.outputDir);
  process.env.BB_PROOF_BENCH_DEFER_PROFILE_WRITE = '1';
  if (args.persistentBbWorker) {
    process.env.BB_PROOF_BENCH_PERSISTENT_BB = '1';
  }
  await mkdir(args.outputDir, { recursive: true });

  const jobs = await discoverProofInputs(args);
  if (jobs.length === 0) {
    throw new Error(`No proof inputs discovered in ${args.proofStore}`);
  }

  await writeFile(join(args.outputDir, 'metadata.json'), JSON.stringify(collectMetadata(args, jobs), null, 2) + '\n');

  if (args.list) {
    for (const [index, job] of jobs.entries()) {
      console.log(`${index.toString().padStart(2, '0')} ${job.typeName} ${basename(job.inputPath)} ${job.inputSha256}`);
    }
    return;
  }

  const proofStore = await createProofStore(args.proofStore, logger);
  const rawPath = join(args.outputDir, 'raw.jsonl');
  await writeFile(rawPath, '');

  const suiteRecords = [];
  const jobRecords = [];
  for (let i = 0; i < args.warmups; i++) {
    const { suiteRecord, records } = await runSuite(args, proofStore, jobs, i, true, rawPath);
    suiteRecords.push(suiteRecord);
    jobRecords.push(...records);
  }
  for (let i = 0; i < args.repeats; i++) {
    const { suiteRecord, records } = await runSuite(args, proofStore, jobs, i, false, rawPath);
    suiteRecords.push(suiteRecord);
    jobRecords.push(...records);
  }

  const summary = summarize(args, jobs, suiteRecords, jobRecords);
  await writeFile(join(args.outputDir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  await writeSummaryMd(args.outputDir, summary);

  console.log(`proof-store replay avg: ${summary.elapsedMsAvg.toFixed(3)} ms over ${summary.samples} sample(s)`);
  console.log(`summary: ${join(args.outputDir, 'summary.md')}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
