#!/usr/bin/env node
import { BBNativeRollupProver } from '@aztec/bb-prover';
import { createLogger } from '@aztec/foundation/log';
import { createProofStore } from '@aztec/prover-client/broker';
import { ProvingRequestType } from '@aztec/stdlib/proofs';

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { gzipSync } from 'node:zlib';

const logger = createLogger('gpu-proof-generation-bench');

function parseArgs() {
  const args = {
    repeats: 1,
    warmups: 0,
    expectedType: 'ROOT_ROLLUP',
    proofStore: undefined,
  };
  for (let i = 2; i < process.argv.length; i++) {
    const key = process.argv[i];
    const value = process.argv[i + 1];
    switch (key) {
      case '--proof-uri':
        args.proofUri = value;
        i++;
        break;
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
      case '--expected-type':
        args.expectedType = value;
        i++;
        break;
      default:
        throw new Error(`Unknown argument: ${key}`);
    }
  }
  for (const key of ['proofUri', 'bbBin', 'acvmBin', 'outputDir']) {
    if (!args[key]) {
      throw new Error(`Missing required argument --${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`);
    }
  }
  return args;
}

function inferProofStore(proofUri) {
  const url = new URL(proofUri);
  if (url.protocol === 'file:') {
    return 'file:///';
  }
  if (url.protocol === 'gs:' || url.protocol === 's3:') {
    return `${url.protocol}//${url.host}`;
  }
  throw new Error(`Unable to infer proof store for ${proofUri}; pass --proof-store explicitly.`);
}

async function maybeFileStats(path) {
  try {
    const data = await readFile(path);
    return {
      bytes: data.length,
      gzipBytes: gzipSync(data).length,
    };
  } catch {
    return {
      bytes: null,
      gzipBytes: null,
    };
  }
}

async function dispatchProof(prover, type, inputs) {
  switch (type) {
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
      throw new Error(`Unsupported proof type for this benchmark: ${ProvingRequestType[type] ?? type}`);
  }
}

async function loadProofJob(proofStore, args) {
  const job = await proofStore.getProofInput(args.proofUri);
  const expectedType = ProvingRequestType[args.expectedType];
  if (expectedType === undefined) {
    throw new Error(`Unknown expected proof type: ${args.expectedType}`);
  }
  if (job.type !== expectedType) {
    throw new Error(`Proof input is ${ProvingRequestType[job.type]}, expected ${args.expectedType}`);
  }
  return job;
}

async function runOne(args, proofStore, repeat, warmup) {
  const start = performance.now();
  const runDir = resolve(
    args.outputDir,
    warmup ? `warmup_${repeat.toString().padStart(2, '0')}` : `run_${repeat.toString().padStart(2, '0')}`,
  );
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
  const setupMs = performance.now() - setupStart;

  const inputLoadStart = performance.now();
  const job = await loadProofJob(proofStore, args);
  const inputLoadMs = performance.now() - inputLoadStart;

  const proofGenerationStart = performance.now();
  const result = await dispatchProof(prover, job.type, job.inputs);
  const proofGenerationMs = performance.now() - proofGenerationStart;

  const proofOutputStart = performance.now();
  const proofPath = join(bbDir, 'proof');
  const proofStats = await maybeFileStats(proofPath);
  const proofOutputMs = performance.now() - proofOutputStart;
  const elapsedMs = performance.now() - start;
  const stageTotalMs = setupMs + inputLoadMs + proofGenerationMs + proofOutputMs;
  const record = {
    benchmark: 'rollup-proof-job',
    proofType: ProvingRequestType[job.type],
    repeat,
    warmup,
    elapsedMs,
    setupMs,
    inputLoadMs,
    proofGenerationMs,
    proofOutputMs,
    stageTotalMs,
    overheadMs: elapsedMs - stageTotalMs,
    proofSizeBytes: proofStats.bytes ?? result?.proof?.binaryProof?.buffer?.length ?? null,
    proofGzipSizeBytes: proofStats.gzipBytes,
    proofPath,
    runDir,
    bbDir,
    acvmDir,
    bbBin: args.bbBin,
    acvmBin: args.acvmBin,
    proofUri: args.proofUri,
  };
  await writeFile(join(runDir, 'record.json'), JSON.stringify(record, null, 2) + '\n');
  return record;
}

function summarize(outputDir, records) {
  const measured = records.filter(record => !record.warmup);
  const elapsed = measured.map(record => record.elapsedMs);
  const avgField = field => measured.reduce((sum, record) => sum + record[field], 0) / measured.length;
  const avg = elapsed.reduce((sum, value) => sum + value, 0) / elapsed.length;
  const variance = elapsed.reduce((sum, value) => sum + (value - avg) ** 2, 0) / elapsed.length;
  return {
    benchmark: 'rollup-proof-job',
    samples: measured.length,
    elapsedMsAvg: avg,
    elapsedMsMin: Math.min(...elapsed),
    elapsedMsMax: Math.max(...elapsed),
    elapsedMsStdev: Math.sqrt(variance),
    setupMsAvg: avgField('setupMs'),
    inputLoadMsAvg: avgField('inputLoadMs'),
    proofGenerationMsAvg: avgField('proofGenerationMs'),
    proofOutputMsAvg: avgField('proofOutputMs'),
    stageTotalMsAvg: avgField('stageTotalMs'),
    overheadMsAvg: avgField('overheadMs'),
    outputDir,
    records: measured,
  };
}

async function main() {
  const args = parseArgs();
  args.outputDir = resolve(args.outputDir);
  await mkdir(args.outputDir, { recursive: true });
  const proofStoreConfig = args.proofStore ?? inferProofStore(args.proofUri);
  const proofStore = await createProofStore(proofStoreConfig, logger);

  const records = [];
  const rawPath = join(args.outputDir, 'raw.jsonl');
  await writeFile(rawPath, '');
  for (let i = 0; i < args.warmups; i++) {
    const record = await runOne(args, proofStore, i, true);
    records.push(record);
    await writeFile(rawPath, JSON.stringify(record) + '\n', { flag: 'a' });
  }
  for (let i = 0; i < args.repeats; i++) {
    const record = await runOne(args, proofStore, i, false);
    records.push(record);
    await writeFile(rawPath, JSON.stringify(record) + '\n', { flag: 'a' });
  }

  const summary = summarize(args.outputDir, records);
  await writeFile(join(args.outputDir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  const lines = [
    '| repeat | proof type | e2e ms | stage total ms | setup ms | input load ms | proof generation ms | output ms | overhead ms | proof bytes | proof gzip bytes | run dir |',
    '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ...summary.records.map(
      record =>
        `| ${record.repeat} | ${record.proofType} | ${record.elapsedMs.toFixed(
          3,
        )} | ${record.stageTotalMs.toFixed(3)} | ${record.setupMs.toFixed(3)} | ${record.inputLoadMs.toFixed(
          3,
        )} | ${record.proofGenerationMs.toFixed(3)} | ${record.proofOutputMs.toFixed(3)} | ${record.overheadMs.toFixed(
          3,
        )} | ${record.proofSizeBytes ?? ''} | ${record.proofGzipSizeBytes ?? ''} | ${record.runDir} |`,
    ),
    '',
  ];
  await writeFile(join(args.outputDir, 'summary.md'), lines.join('\n'));
  console.log(`elapsed avg: ${summary.elapsedMsAvg.toFixed(3)} ms over ${summary.samples} sample(s)`);
  console.log(`summary: ${join(args.outputDir, 'summary.md')}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
