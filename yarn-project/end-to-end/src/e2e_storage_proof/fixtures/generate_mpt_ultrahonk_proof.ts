import { Barretenberg, UltraHonkBackend, deflattenFields } from '@aztec/bb.js';
import { Noir } from '@aztec/noir-noir_js';

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const STORAGE_PROOF_FIXTURE_PATH = join(__dirname, './storage_proof.json');
const OUTPUT_PATH = join(__dirname, './storage_proof_ultrahonk.json');
const DEFAULT_ARTIFACT_PATH = resolve(
  __dirname,
  '../../../../../noir-projects/noir-protocol-circuits/target/storage_proof_mpt.json',
);
const CIRCUIT_ARTIFACT_PATH = process.env.STORAGE_PROOF_MPT_ARTIFACT ?? DEFAULT_ARTIFACT_PATH;

type JsonNode = { rows: string[][]; row_exist: boolean[]; node_type: string };

type JsonAccount = {
  nonce: string[];
  balance: string[];
  address: string[];
  nonce_length: string;
  balance_length: string;
  storage_hash: string[];
  code_hash: string[];
};

type StorageProofJSON = {
  root: string[];
  slot_key: string[];
  account_node_length: string;
  storage_node_length: string;
  account_nodes: JsonNode[];
  storage_nodes: JsonNode[];
  account: JsonAccount;
  slot: { value: string[]; value_length: string };
};

const fixture: StorageProofJSON = JSON.parse(readFileSync(STORAGE_PROOF_FIXTURE_PATH, 'utf8'));
const circuitJson = JSON.parse(readFileSync(CIRCUIT_ARTIFACT_PATH, 'utf8'));

const input = {
  account: fixture.account,
  account_nodes: fixture.account_nodes,
  account_node_length: fixture.account_node_length,
  storage_nodes: fixture.storage_nodes,
  storage_node_length: fixture.storage_node_length,
  public_inputs: {
    state_root: fixture.root,
    address: fixture.account.address,
    slot_key: fixture.slot_key,
    value: fixture.slot.value,
    value_length: fixture.slot.value_length,
  },
};

const barretenbergAPI = await Barretenberg.new({ threads: Number(process.env.BB_THREADS ?? 1) });
const circuit = new Noir(circuitJson);
const backend = new UltraHonkBackend(circuitJson.bytecode, barretenbergAPI);

const witnessStart = performance.now();
const { witness } = await circuit.execute(input);
const witnessMs = performance.now() - witnessStart;

const proveStart = performance.now();
const proofData = await backend.generateProof(witness, { verifierTarget: 'noir-recursive' });
const proveMs = performance.now() - proveStart;

const verifyStart = performance.now();
const valid = await backend.verifyProof(proofData, { verifierTarget: 'noir-recursive' });
const verifyMs = performance.now() - verifyStart;
if (!valid) {
  throw new Error('Generated storage MPT proof did not verify locally');
}

const tamperedPublicInputs = [...proofData.publicInputs];
const tamperedValue = BigInt(tamperedPublicInputs[tamperedPublicInputs.length - 1]) + 1n;
tamperedPublicInputs[tamperedPublicInputs.length - 1] = `0x${tamperedValue.toString(16).padStart(64, '0')}`;
const tamperedValid = await backend.verifyProof(
  { ...proofData, publicInputs: tamperedPublicInputs },
  { verifierTarget: 'noir-recursive' },
);
if (tamperedValid) {
  throw new Error('Generated storage MPT proof verified after tampering with public inputs');
}

const recursiveArtifacts = await backend.generateRecursiveProofArtifacts(
  proofData.proof,
  proofData.publicInputs.length,
);

let proofAsFields = recursiveArtifacts.proofAsFields;
if (proofAsFields.length === 0) {
  proofAsFields = deflattenFields(proofData.proof).map(field => field.toString());
}

const output = {
  vkAsFields: recursiveArtifacts.vkAsFields,
  vkHash: recursiveArtifacts.vkHash,
  proofAsFields,
  publicInputs: proofData.publicInputs.map(publicInput => publicInput.toString()),
  timings: {
    witnessMs,
    proveMs,
    verifyMs,
  },
  sizes: {
    vkFields: recursiveArtifacts.vkAsFields.length,
    proofFields: proofAsFields.length,
    publicInputs: proofData.publicInputs.length,
  },
};

writeFileSync(OUTPUT_PATH, JSON.stringify(output, null, 2));
await barretenbergAPI.destroy();

console.log(`Wrote ${OUTPUT_PATH}`);
console.log(JSON.stringify({ timings: output.timings, sizes: output.sizes }, null, 2));
