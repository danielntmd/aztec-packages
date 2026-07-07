import { sha256 } from '@aztec/foundation/crypto/sha256';
import type { LogFn, Logger } from '@aztec/foundation/log';
import { Timer } from '@aztec/foundation/timer';
import type { AvmCircuitInputs, AvmCircuitPublicInputs } from '@aztec/stdlib/avm';

import * as proc from 'child_process';
import { promises as fs } from 'fs';
import { basename, dirname, join } from 'path';
import readline from 'readline';

import type { UltraHonkFlavor } from '../honk.js';

export const VK_FILENAME = 'vk';
export const PUBLIC_INPUTS_FILENAME = 'public_inputs';
export const PROOF_FILENAME = 'proof';
export const BB_BENCH_HIERARCHICAL_FILENAME = 'bb-bench-hierarchical.json';
export const AVM_INPUTS_FILENAME = 'avm_inputs.bin';
export const AVM_BYTECODE_FILENAME = 'avm_bytecode.bin';
export const AVM_PUBLIC_INPUTS_FILENAME = 'avm_public_inputs.bin';

export enum BB_RESULT {
  SUCCESS,
  FAILURE,
  ALREADY_PRESENT,
}

export type BBSuccess = {
  status: BB_RESULT.SUCCESS | BB_RESULT.ALREADY_PRESENT;
  durationMs: number;
  /** Full path of the public key. */
  pkPath?: string;
  /** Base directory for the VKs (raw, fields). */
  vkDirectoryPath?: string;
  /** Full path of the proof. */
  proofPath?: string;
  /** Full path of the hierarchical BB benchmark JSON, when BB_PROOF_BENCH=1. */
  benchPath?: string;
  /** Full path of the contract. */
  contractPath?: string;
  /** The number of gates in the circuit. */
  circuitSize?: number;
};

export type BBFailure = {
  status: BB_RESULT.FAILURE;
  reason: string;
  retry?: boolean;
};

export type BBResult = BBSuccess | BBFailure;

type BBExecResult = {
  status: BB_RESULT;
  exitCode: number;
  signal: string | undefined;
};

type BBWorkerResponse = {
  id: string;
  status: 'ok' | 'error';
  reason?: string;
  duration_ms?: number;
  prewarm_ms?: number;
  bench_out_hierarchical?: string;
};

type CircuitAssetPaths = {
  bytecodePath: string;
  vkPath: string;
};

function envFlag(name: string) {
  const value = process.env[name]?.toLowerCase();
  return value === '1' || value === 'true';
}

function proofMemorySettings() {
  return {
    slowLowMemory: envFlag('BB_SLOW_LOW_MEMORY'),
    storageBudget: process.env.BB_STORAGE_BUDGET,
  };
}

function addProofMemoryArgs(args: string[]) {
  const settings = proofMemorySettings();
  if (settings.slowLowMemory) {
    args.push('--slow_low_memory');
  }
  if (settings.storageBudget) {
    args.push('--storage_budget', settings.storageBudget);
  }
}

export const DEFAULT_BB_VERIFY_CONCURRENCY = 4;

/**
 * Invokes the Barretenberg binary with the provided command and args
 * @param pathToBB - The path to the BB binary
 * @param command - The command to execute
 * @param args - The arguments to pass
 * @param logger - A log function
 * @param timeout - An optional timeout before killing the BB process
 * @param resultParser - An optional handler for detecting success or failure
 * @returns The completed partial witness outputted from the circuit
 */
export function executeBB(
  pathToBB: string,
  command: string,
  args: string[],
  logger: LogFn,
  concurrency?: number,
  timeout?: number,
  resultParser = (code: number) => code === 0,
): Promise<BBExecResult> {
  return new Promise<BBExecResult>(resolve => {
    // spawn the bb process
    const { HARDWARE_CONCURRENCY: _, ...envWithoutConcurrency } = process.env;

    const env = envWithoutConcurrency;
    // We prioritise the concurrency argument if provided and > 0
    if (concurrency && concurrency > 0) {
      env.HARDWARE_CONCURRENCY = concurrency.toString();
    } else if (process.env.HARDWARE_CONCURRENCY) {
      env.HARDWARE_CONCURRENCY = process.env.HARDWARE_CONCURRENCY;
    }

    logger(`BB concurrency: ${env.HARDWARE_CONCURRENCY}`);
    logger(`Executing BB with: ${pathToBB} ${command} ${args.join(' ')}`);
    const bb = proc.spawn(pathToBB, [command, ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env,
    });

    let timeoutId: NodeJS.Timeout | undefined;
    if (timeout !== undefined) {
      timeoutId = setTimeout(() => {
        logger(`BB execution timed out after ${timeout}ms, killing process`);
        if (bb.pid) {
          bb.kill('SIGKILL');
        }
        resolve({ status: BB_RESULT.FAILURE, exitCode: -1, signal: 'TIMEOUT' });
      }, timeout);
    }

    readline.createInterface({ input: bb.stdout }).on('line', logger);
    readline.createInterface({ input: bb.stderr }).on('line', logger);

    bb.on('close', (exitCode: number, signal?: string) => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (resultParser(exitCode)) {
        resolve({ status: BB_RESULT.SUCCESS, exitCode, signal });
      } else {
        resolve({ status: BB_RESULT.FAILURE, exitCode, signal });
      }
    });
  }).catch(_ => ({ status: BB_RESULT.FAILURE, exitCode: -1, signal: undefined }));
}

export async function executeBbChonkProof(
  pathToBB: string,
  workingDirectory: string,
  inputsPath: string,
  log: LogFn,
  writeVk = false,
): Promise<BBFailure | BBSuccess> {
  // Check that the working directory exists
  try {
    await fs.access(workingDirectory);
  } catch {
    return { status: BB_RESULT.FAILURE, reason: `Working directory ${workingDirectory} does not exist` };
  }

  // The proof is written to e.g. /workingDirectory/proof
  const outputPath = `${workingDirectory}`;

  const binaryPresent = await fs
    .access(pathToBB, fs.constants.R_OK)
    .then(_ => true)
    .catch(_ => false);
  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  try {
    // Write the bytecode to the working directory
    log(`inputsPath ${inputsPath}`);
    const timer = new Timer();
    const logFunction = (message: string) => {
      log(`bb - ${message}`);
    };

    const benchPath =
      process.env.BB_PROOF_BENCH === '1' ? join(workingDirectory, BB_BENCH_HIERARCHICAL_FILENAME) : undefined;
    const args = ['-o', outputPath, '--ivc_inputs_path', inputsPath, '-v', '--scheme', 'chonk'];
    if (writeVk) {
      args.push('--write_vk');
    }
    if (benchPath) {
      args.push('--bench_out_hierarchical', benchPath);
    }
    const result = await executeBB(pathToBB, 'prove', args, logFunction);
    const durationMs = timer.ms();

    if (result.status == BB_RESULT.SUCCESS) {
      return {
        status: BB_RESULT.SUCCESS,
        durationMs,
        proofPath: `${outputPath}`,
        benchPath,
        pkPath: undefined,
        vkDirectoryPath: `${outputPath}`,
      };
    }
    // Not a great error message here but it is difficult to decipher what comes from bb
    return {
      status: BB_RESULT.FAILURE,
      reason: `Failed to generate proof. Exit code ${result.exitCode}. Signal ${result.signal}.`,
      retry: !!result.signal,
    };
  } catch (error) {
    return { status: BB_RESULT.FAILURE, reason: `${error}` };
  }
}

function getArgs(flavor: UltraHonkFlavor) {
  switch (flavor) {
    case 'ultra_honk': {
      return ['--scheme', 'ultra_honk', '--oracle_hash', 'poseidon2'];
    }
    case 'ultra_keccak_honk': {
      return ['--scheme', 'ultra_honk', '--oracle_hash', 'keccak'];
    }
    case 'ultra_starknet_honk': {
      return ['--scheme', 'ultra_honk', '--oracle_hash', 'starknet'];
    }
    case 'ultra_rollup_honk': {
      return ['--scheme', 'ultra_honk', '--oracle_hash', 'poseidon2', '--ipa_accumulation'];
    }
  }
}

function getWorkerSettings(flavor: UltraHonkFlavor) {
  switch (flavor) {
    case 'ultra_honk':
      return { oracleHashType: 'poseidon2', ipaAccumulation: false };
    case 'ultra_keccak_honk':
      return { oracleHashType: 'keccak', ipaAccumulation: false };
    case 'ultra_starknet_honk':
      return { oracleHashType: 'starknet', ipaAccumulation: false };
    case 'ultra_rollup_honk':
      return { oracleHashType: 'poseidon2', ipaAccumulation: true };
  }
}

export class UltraHonkProveWorker {
  private child?: proc.ChildProcessWithoutNullStreams;
  private nextRequestId = 0;
  private readonly pending = new Map<string, { resolve: (response: BBWorkerResponse) => void; reject: (err: Error) => void }>();
  private readonly circuitAssets = new Map<string, Promise<CircuitAssetPaths>>();

  constructor(
    private readonly pathToBB: string,
    private readonly log: LogFn,
  ) {}

  async prewarmSrs(numPoints: number): Promise<number> {
    if (numPoints <= 0) {
      return 0;
    }
    const response = await this.send({ type: 'prewarm_srs', num_points: numPoints });
    return response.prewarm_ms ?? 0;
  }

  async prepareCircuit(
    assetRootDirectory: string,
    circuitName: string,
    bytecode: Buffer,
    verificationKey: Buffer,
  ): Promise<number> {
    const assetDirectory = join(assetRootDirectory, 'worker-assets');
    const key = this.circuitAssetKey(assetDirectory, circuitName);
    if (this.circuitAssets.has(key)) {
      await this.circuitAssets.get(key);
      return 0;
    }

    const timer = new Timer();
    const writePromise = this.writeCircuitAssets(assetDirectory, circuitName, bytecode, verificationKey).catch(error => {
      this.circuitAssets.delete(key);
      throw error;
    });
    this.circuitAssets.set(key, writePromise);
    await writePromise;
    return timer.ms();
  }

  async prove(
    workingDirectory: string,
    circuitName: string,
    bytecode: Buffer,
    verificationKey: Buffer,
    inputWitnessFile: string,
    flavor: UltraHonkFlavor,
  ): Promise<BBFailure | BBSuccess> {
    try {
      await fs.access(workingDirectory);
    } catch {
      return { status: BB_RESULT.FAILURE, reason: `Working directory ${workingDirectory} does not exist` };
    }

    const binaryPresent = await fs
      .access(this.pathToBB, fs.constants.R_OK)
      .then(_ => true)
      .catch(_ => false);
    if (!binaryPresent) {
      return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${this.pathToBB}` };
    }

    const { bytecodePath, vkPath } = await this.getCircuitAssets(
      dirname(workingDirectory),
      circuitName,
      bytecode,
      verificationKey,
    );
    const outputPath = `${workingDirectory}`;
    const benchPath =
      process.env.BB_PROOF_BENCH === '1' ? join(workingDirectory, BB_BENCH_HIERARCHICAL_FILENAME) : undefined;
    const settings = getWorkerSettings(flavor);
    const memorySettings = proofMemorySettings();

    try {
      const timer = new Timer();
      const response = await this.send({
        type: 'prove',
        oracle_hash_type: settings.oracleHashType,
        ipa_accumulation: settings.ipaAccumulation,
        disable_zk: true,
        write_vk: false,
        output_format: 'binary',
        bytecode_path: bytecodePath,
        witness_path: inputWitnessFile,
        vk_path: vkPath,
        output_path: outputPath,
        slow_low_memory: memorySettings.slowLowMemory,
        ...(memorySettings.storageBudget ? { storage_budget: memorySettings.storageBudget } : {}),
        ...(benchPath ? { bench_out_hierarchical: benchPath } : {}),
      });
      return {
        status: BB_RESULT.SUCCESS,
        durationMs: timer.ms(),
        proofPath: `${outputPath}`,
        benchPath: response.bench_out_hierarchical ?? benchPath,
        pkPath: undefined,
        vkDirectoryPath: `${outputPath}`,
      };
    } catch (error) {
      return { status: BB_RESULT.FAILURE, reason: `${error}`, retry: true };
    }
  }

  private async getCircuitAssets(
    assetRootDirectory: string,
    circuitName: string,
    bytecode: Buffer,
    verificationKey: Buffer,
  ): Promise<CircuitAssetPaths> {
    await this.prepareCircuit(assetRootDirectory, circuitName, bytecode, verificationKey);
    return await this.circuitAssets.get(this.circuitAssetKey(join(assetRootDirectory, 'worker-assets'), circuitName))!;
  }

  private circuitAssetKey(assetDirectory: string, circuitName: string) {
    return `${assetDirectory}/${circuitName}`;
  }

  private async writeCircuitAssets(
    assetDirectory: string,
    circuitName: string,
    bytecode: Buffer,
    verificationKey: Buffer,
  ): Promise<CircuitAssetPaths> {
    await fs.mkdir(assetDirectory, { recursive: true });
    const bytecodePath = `${assetDirectory}/${circuitName}-bytecode`;
    const vkPath = `${assetDirectory}/${circuitName}-vk`;
    await Promise.all([fs.writeFile(bytecodePath, bytecode), fs.writeFile(vkPath, verificationKey)]);
    return { bytecodePath, vkPath };
  }

  async stop(): Promise<void> {
    if (!this.child) {
      return;
    }
    try {
      await this.send({ type: 'shutdown' });
    } catch {
      this.child.kill('SIGKILL');
    }
  }

  private ensureStarted() {
    if (this.child) {
      return;
    }

    const { HARDWARE_CONCURRENCY: _, BB_GPU_MSM_PREWARM_SRS_POINTS: __, ...env } = process.env;
    if (process.env.HARDWARE_CONCURRENCY) {
      env.HARDWARE_CONCURRENCY = process.env.HARDWARE_CONCURRENCY;
    }

    this.log(`Executing persistent BB worker with: ${this.pathToBB} prove_ultra_honk_worker`);
    this.child = proc.spawn(this.pathToBB, ['prove_ultra_honk_worker'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
    });

    readline.createInterface({ input: this.child.stdout }).on('line', line => {
      const prefix = 'BB_WORKER_RESULT ';
      if (!line.startsWith(prefix)) {
        this.log(line);
        return;
      }
      let response: BBWorkerResponse;
      try {
        response = JSON.parse(line.slice(prefix.length));
      } catch (err) {
        this.log(`Failed to parse BB worker response: ${err}`);
        return;
      }
      const pending = this.pending.get(response.id);
      if (!pending) {
        return;
      }
      this.pending.delete(response.id);
      if (response.status === 'ok') {
        pending.resolve(response);
      } else {
        pending.reject(new Error(response.reason ?? 'BB worker request failed'));
      }
    });
    readline.createInterface({ input: this.child.stderr }).on('line', this.log);
    this.child.on('close', (code, signal) => {
      const err = new Error(`BB worker exited with code ${code} signal ${signal}`);
      for (const pending of this.pending.values()) {
        pending.reject(err);
      }
      this.pending.clear();
      this.child = undefined;
    });
  }

  private send(request: Record<string, unknown>): Promise<BBWorkerResponse> {
    this.ensureStarted();
    return new Promise<BBWorkerResponse>((resolve, reject) => {
      const id = `${++this.nextRequestId}`;
      this.pending.set(id, { resolve, reject });
      this.child!.stdin.write(`${JSON.stringify({ id, ...request })}\n`, err => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }
}

export async function generateProofWithWorker(
  worker: UltraHonkProveWorker,
  workingDirectory: string,
  circuitName: string,
  bytecode: Buffer,
  verificationKey: Buffer,
  inputWitnessFile: string,
  flavor: UltraHonkFlavor,
): Promise<BBFailure | BBSuccess> {
  return await worker.prove(workingDirectory, circuitName, bytecode, verificationKey, inputWitnessFile, flavor);
}

/**
 * Used for generating proofs of noir circuits.
 * It is assumed that the working directory is a temporary and/or random directory used solely for generating this proof.
 * @param pathToBB - The full path to the bb binary
 * @param workingDirectory - A working directory for use by bb
 * @param circuitName - An identifier for the circuit
 * @param bytecode - The compiled circuit bytecode
 * @param inputWitnessFile - The circuit input witness
 * @param log - A logging function
 * @returns An object containing a result indication, the location of the proof and the duration taken
 */
export async function generateProof(
  pathToBB: string,
  workingDirectory: string,
  circuitName: string,
  bytecode: Buffer,
  verificationKey: Buffer,
  inputWitnessFile: string,
  flavor: UltraHonkFlavor,
  log: Logger,
): Promise<BBFailure | BBSuccess> {
  // Check that the working directory exists
  try {
    await fs.access(workingDirectory);
  } catch {
    return { status: BB_RESULT.FAILURE, reason: `Working directory ${workingDirectory} does not exist` };
  }

  // The bytecode is written to e.g. /workingDirectory/ParityBaseArtifact-bytecode
  const bytecodePath = `${workingDirectory}/${circuitName}-bytecode`;
  const vkPath = `${workingDirectory}/${circuitName}-vk`;

  // The proof is written to e.g. /workingDirectory/ultra_honk/proof
  const outputPath = `${workingDirectory}`;

  const binaryPresent = await fs
    .access(pathToBB, fs.constants.R_OK)
    .then(_ => true)
    .catch(_ => false);
  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  try {
    // Write the bytecode and vk to the working directory
    await Promise.all([fs.writeFile(bytecodePath, bytecode), fs.writeFile(vkPath, verificationKey)]);
    const args = getArgs(flavor).concat([
      '--disable_zk',
      '-o',
      outputPath,
      '-b',
      bytecodePath,
      '-k',
      vkPath,
      '-w',
      inputWitnessFile,
      '-v',
    ]);
    const benchPath =
      process.env.BB_PROOF_BENCH === '1' ? join(workingDirectory, BB_BENCH_HIERARCHICAL_FILENAME) : undefined;
    if (benchPath) {
      args.push('--bench_out_hierarchical', benchPath);
    }
    addProofMemoryArgs(args);
    const loggingArg = log.level === 'debug' || log.level === 'trace' ? '-d' : log.level === 'verbose' ? '-v' : '';
    if (loggingArg !== '') {
      args.push(loggingArg);
    }

    const timer = new Timer();
    const logFunction = (message: string) => {
      log.info(`${circuitName} BB out - ${message}`);
    };
    const result = await executeBB(pathToBB, `prove`, args, logFunction);
    const duration = timer.ms();

    if (result.status == BB_RESULT.SUCCESS) {
      return {
        status: BB_RESULT.SUCCESS,
        durationMs: duration,
        proofPath: `${outputPath}`,
        benchPath,
        pkPath: undefined,
        vkDirectoryPath: `${outputPath}`,
      };
    }
    // Not a great error message here but it is difficult to decipher what comes from bb
    return {
      status: BB_RESULT.FAILURE,
      reason: `Failed to generate proof. Exit code ${result.exitCode}. Signal ${result.signal}.`,
      retry: !!result.signal,
    };
  } catch (error) {
    return { status: BB_RESULT.FAILURE, reason: `${error}` };
  }
}

/**
 * Used for generating AVM proofs.
 * It is assumed that the working directory is a temporary and/or random directory used solely for generating this proof.
 * @param pathToBB - The full path to the bb binary
 * @param workingDirectory - A working directory for use by bb
 * @param input - The inputs for the public function to be proven
 * @param logger - A logging function
 * @param checkCircuitOnly - A boolean to toggle a "check-circuit only" operation instead of proving.
 * @returns An object containing a result indication, the location of the proof and the duration taken
 */
export async function generateAvmProof(
  pathToBB: string,
  workingDirectory: string,
  input: AvmCircuitInputs,
  logger: Logger,
  checkCircuitOnly: boolean = false,
): Promise<BBFailure | BBSuccess> {
  // Check that the working directory exists
  try {
    await fs.access(workingDirectory);
  } catch {
    return { status: BB_RESULT.FAILURE, reason: `Working directory ${workingDirectory} does not exist` };
  }

  // The proof is written to e.g. /workingDirectory/proof
  const outputPath = workingDirectory;

  const filePresent = async (file: string) =>
    await fs
      .access(file, fs.constants.R_OK)
      .then(_ => true)
      .catch(_ => false);

  const binaryPresent = await filePresent(pathToBB);
  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  const inputsBuffer = input.serializeWithMessagePack();

  try {
    // Write the inputs to the working directory.
    const avmInputsPath = join(workingDirectory, AVM_INPUTS_FILENAME);
    await fs.writeFile(avmInputsPath, inputsBuffer);
    if (!(await filePresent(avmInputsPath))) {
      return { status: BB_RESULT.FAILURE, reason: `Could not write avm inputs to ${avmInputsPath}` };
    }

    const args = checkCircuitOnly ? ['--avm-inputs', avmInputsPath] : ['--avm-inputs', avmInputsPath, '-o', outputPath];
    const loggingArg =
      logger.level === 'debug' || logger.level === 'trace' ? '-d' : logger.level === 'verbose' ? '-v' : '';
    if (loggingArg !== '') {
      args.push(loggingArg);
    }
    const timer = new Timer();

    const cmd = checkCircuitOnly ? 'avm_check_circuit' : 'avm_prove';
    const logFunction = (message: string) => {
      logger.verbose(`AvmCircuit (${cmd}) BB out - ${message}`);
    };
    const result = await executeBB(pathToBB, cmd, args, logFunction);
    const duration = timer.ms();

    if (result.status == BB_RESULT.SUCCESS) {
      return {
        status: BB_RESULT.SUCCESS,
        durationMs: duration,
        proofPath: join(outputPath, PROOF_FILENAME),
        pkPath: undefined,
        vkDirectoryPath: undefined, // AVM VK is fixed in the binary.
      };
    }
    // Not a great error message here but it is difficult to decipher what comes from bb
    return {
      status: BB_RESULT.FAILURE,
      reason: `Failed to generate proof. AVM proof for TX hash ${input.hints.tx.hash}. Exit code ${result.exitCode}. Signal ${result.signal}.`,
      retry: result.signal === 'SIGKILL', // retry on SIGKILL because the oomkiller might have stopped the process
    };
  } catch (error) {
    return { status: BB_RESULT.FAILURE, reason: `${error}` };
  }
}

/**
 * Used for verifying proofs of noir circuits
 * @param pathToBB - The full path to the bb binary
 * @param proofFullPath - The full path to the proof to be verified
 * @param verificationKeyPath - The full path to the circuit verification key
 * @param logger - A logger
 * @returns An object containing a result indication and duration taken
 */
export async function verifyProof(
  pathToBB: string,
  proofFullPath: string,
  verificationKeyPath: string,
  ultraHonkFlavor: UltraHonkFlavor,
  logger: Logger,
): Promise<BBFailure | BBSuccess> {
  // Specify the public inputs path in the case of UH verification.
  // Take proofFullPath and remove the suffix past the / to get the directory.
  const proofDir = proofFullPath.substring(0, proofFullPath.lastIndexOf('/'));
  const publicInputsFullPath = join(proofDir, '/public_inputs');
  logger.debug(`public inputs path: ${publicInputsFullPath}`);

  const args = [
    '-p',
    proofFullPath,
    '-k',
    verificationKeyPath,
    '-i',
    publicInputsFullPath,
    '--disable_zk',
    ...getArgs(ultraHonkFlavor),
  ];

  let concurrency = DEFAULT_BB_VERIFY_CONCURRENCY;

  if (process.env.VERIFY_HARDWARE_CONCURRENCY) {
    concurrency = parseInt(process.env.VERIFY_HARDWARE_CONCURRENCY, 10);
  }

  return await verifyProofInternal(pathToBB, `verify`, args, logger, concurrency);
}

export async function verifyAvmProof(
  pathToBB: string,
  workingDirectory: string,
  proofFullPath: string,
  publicInputs: AvmCircuitPublicInputs,
  logger: Logger,
): Promise<BBFailure | BBSuccess> {
  const inputsBuffer = publicInputs.serializeWithMessagePack();

  // Write the inputs to the working directory.
  const filePresent = async (file: string) =>
    await fs
      .access(file, fs.constants.R_OK)
      .then(_ => true)
      .catch(_ => false);
  const avmInputsPath = join(workingDirectory, 'avm_public_inputs.bin');
  await fs.writeFile(avmInputsPath, inputsBuffer);
  if (!(await filePresent(avmInputsPath))) {
    return { status: BB_RESULT.FAILURE, reason: `Could not write avm inputs to ${avmInputsPath}` };
  }

  const args = ['-p', proofFullPath, '--avm-public-inputs', avmInputsPath];
  return await verifyProofInternal(pathToBB, 'avm_verify', args, logger);
}

/**
 * Verifies a ChonkProof
 * TODO(#7370) The verification keys should be supplied separately
 * @param pathToBB - The full path to the bb binary
 * @param targetPath - The path to the folder with the proof, accumulator, and verification keys
 * @param logger - A logger
 * @param concurrency - The number of threads to use for the verification
 * @returns An object containing a result indication and duration taken
 */
export async function verifyChonkProof(
  pathToBB: string,
  proofPath: string,
  keyPath: string,
  logger: Logger,
  concurrency = 1,
): Promise<BBFailure | BBSuccess> {
  const binaryPresent = await fs
    .access(pathToBB, fs.constants.R_OK)
    .then(_ => true)
    .catch(_ => false);
  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  const args = ['--scheme', 'chonk', '-p', proofPath, '-k', keyPath, '-v'];
  return await verifyProofInternal(pathToBB, 'verify', args, logger, concurrency);
}

/**
 * Used for verifying proofs with BB
 * @param pathToBB - The full path to the bb binary
 * @param command - The BB command to execute (verify/avm_verify)
 * @param args - The arguments to pass to the command
 * @param logger - A logger
 * @param concurrency - The number of threads to use for the verification
 * @returns An object containing a result indication and duration taken
 */
async function verifyProofInternal(
  pathToBB: string,
  command: 'verify' | 'avm_verify',
  args: string[],
  logger: Logger,
  concurrency?: number,
): Promise<BBFailure | BBSuccess> {
  const binaryPresent = await fs
    .access(pathToBB, fs.constants.R_OK)
    .then(_ => true)
    .catch(_ => false);
  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  const logFunction = (message: string) => {
    logger.verbose(`bb-prover (verify) BB out - ${message}`);
  };

  try {
    const loggingArg =
      logger.level === 'debug' || logger.level === 'trace' ? '-d' : logger.level === 'verbose' ? '-v' : '';
    const finalArgs = loggingArg !== '' ? [...args, loggingArg] : args;

    const timer = new Timer();
    const result = await executeBB(pathToBB, command, finalArgs, logFunction, concurrency);
    const duration = timer.ms();
    if (result.status == BB_RESULT.SUCCESS) {
      return { status: BB_RESULT.SUCCESS, durationMs: duration };
    }
    // Not a great error message here but it is difficult to decipher what comes from bb
    return {
      status: BB_RESULT.FAILURE,
      reason: `Failed to verify proof. Exit code ${result.exitCode}. Signal ${result.signal}.`,
      retry: !!result.signal,
    };
  } catch (error) {
    return { status: BB_RESULT.FAILURE, reason: `${error}` };
  }
}

export async function generateContractForVerificationKey(
  pathToBB: string,
  vkFilePath: string,
  contractPath: string,
  log: LogFn,
): Promise<BBFailure | BBSuccess> {
  const binaryPresent = await fs
    .access(pathToBB, fs.constants.R_OK)
    .then(_ => true)
    .catch(_ => false);

  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  const outputDir = dirname(contractPath);
  const contractName = basename(contractPath);
  // cache contract generation based on vk file and contract name
  const cacheKey = sha256(Buffer.concat([Buffer.from(contractName), await fs.readFile(vkFilePath)]));

  await fs.mkdir(outputDir, { recursive: true });

  const res = await fsCache<BBSuccess | BBFailure>(outputDir, cacheKey, log, false, async () => {
    try {
      const args = ['--scheme', 'ultra_honk', '-k', vkFilePath, '-o', contractPath, '-v'];
      const timer = new Timer();
      const result = await executeBB(pathToBB, 'contract', args, log);
      const duration = timer.ms();
      if (result.status == BB_RESULT.SUCCESS) {
        return { status: BB_RESULT.SUCCESS, durationMs: duration, contractPath };
      }
      // Not a great error message here but it is difficult to decipher what comes from bb
      return {
        status: BB_RESULT.FAILURE,
        reason: `Failed to write verifier contract. Exit code ${result.exitCode}. Signal ${result.signal}.`,
        retry: !!result.signal,
      };
    } catch (error) {
      return { status: BB_RESULT.FAILURE, reason: `${error}` };
    }
  });

  if (!res) {
    return {
      status: BB_RESULT.ALREADY_PRESENT,
      durationMs: 0,
      contractPath,
    };
  }

  return res;
}

/**
 * Compute bb gate count for a given circuit
 * @param pathToBB - The full path to the bb binary
 * @param workingDirectory - A temporary directory for writing the bytecode
 * @param circuitName - The name of the circuit
 * @param bytecode - The bytecode of the circuit
 * @param flavor - The flavor of the backend - mega_honk or ultra_honk variants
 * @returns An object containing the status, gate count, and time taken
 */
export async function computeGateCountForCircuit(
  pathToBB: string,
  workingDirectory: string,
  circuitName: string,
  bytecode: Buffer,
  flavor: UltraHonkFlavor | 'mega_honk',
  log: LogFn,
): Promise<BBFailure | BBSuccess> {
  // Check that the working directory exists
  try {
    await fs.access(workingDirectory);
  } catch {
    return { status: BB_RESULT.FAILURE, reason: `Working directory ${workingDirectory} does not exist` };
  }

  // The bytecode is written to e.g. /workingDirectory/ParityBaseArtifact-bytecode
  const bytecodePath = `${workingDirectory}/${circuitName}-bytecode`;

  const binaryPresent = await fs
    .access(pathToBB, fs.constants.R_OK)
    .then(_ => true)
    .catch(_ => false);
  if (!binaryPresent) {
    return { status: BB_RESULT.FAILURE, reason: `Failed to find bb binary at ${pathToBB}` };
  }

  // Accumulate the stdout from bb
  let stdout = '';
  const logHandler = (message: string) => {
    stdout += message;
    log(message);
  };

  try {
    // Write the bytecode to the working directory
    await fs.writeFile(bytecodePath, bytecode);
    const timer = new Timer();

    const result = await executeBB(
      pathToBB,
      'gates',
      ['--scheme', flavor === 'mega_honk' ? 'chonk' : 'ultra_honk', '-b', bytecodePath, '-v'],
      logHandler,
    );
    const duration = timer.ms();

    if (result.status == BB_RESULT.SUCCESS) {
      // Look for "circuit_size" in the stdout and parse the number
      const circuitSizeMatch = stdout.match(/circuit_size": (\d+)/);
      if (!circuitSizeMatch) {
        return { status: BB_RESULT.FAILURE, reason: 'Failed to parse circuit_size from bb gates stdout.' };
      }
      const circuitSize = parseInt(circuitSizeMatch[1]);

      return {
        status: BB_RESULT.SUCCESS,
        durationMs: duration,
        circuitSize: circuitSize,
      };
    }

    return { status: BB_RESULT.FAILURE, reason: 'Failed getting the gate count.' };
  } catch (error) {
    return { status: BB_RESULT.FAILURE, reason: `${error}` };
  }
}

const CACHE_FILENAME = '.cache';
async function fsCache<T>(
  dir: string,
  expectedCacheKey: Buffer,
  logger: LogFn,
  force: boolean,
  action: () => Promise<T>,
): Promise<T | undefined> {
  const cacheFilePath = join(dir, CACHE_FILENAME);

  let run: boolean;
  if (force) {
    run = true;
  } else {
    try {
      run = !expectedCacheKey.equals(await fs.readFile(cacheFilePath));
    } catch (err: any) {
      if (err && 'code' in err && err.code === 'ENOENT') {
        // cache file doesn't exist, swallow error and run
        run = true;
      } else {
        throw err;
      }
    }
  }

  let res: T | undefined;
  if (run) {
    logger(`Cache miss or forced run. Running operation in ${dir}...`);
    res = await action();
  } else {
    logger(`Cache hit. Skipping operation in ${dir}...`);
  }

  try {
    await fs.writeFile(cacheFilePath, expectedCacheKey);
  } catch {
    logger(`Couldn't write cache data to ${cacheFilePath}. Skipping cache...`);
    // ignore
  }

  return res;
}
