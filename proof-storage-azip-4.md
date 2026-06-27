# AZIP-4 Storage Proof Offload Benchmark Plan

## Purpose

This note defines a benchmark plan for the AZIP-4 UltraHonk proof offload idea for Ethereum
storage proofs. It is based on the design prompt describing "Offloading the keccak/MPT proof via
recursive UltraHonk verification".

This is specifically the private app-level proof offload path:

```text
standalone MPT proof
  -> recursively verified inside a user's Aztec private transaction
```

It is not the public same-block path, and it is not the in-protocol rollup-recursive variant where
the rollup circuit verifies the MPT proof and exposes the result as canonical block context.

The benchmark is not trying to prove that total system work disappears. The MPT and Keccak work
still has to be proven by someone. The question is narrower:

> Can a user's Aztec private transaction replace direct Ethereum MPT/Keccak verification with a
> much cheaper recursive verification of a standalone UltraHonk proof?

The expected outcome is that the UltraHonk offload path is materially better for the client:

```text
recursive_storage_mpt_verify
  should be much faster and less resource-consuming for the user's PXE
  than
direct_storage_mpt
```

In other words, the benchmark should show that the client-side path pays roughly for recursive proof
verification, not for the full Keccak/MPT relation.

This matters because direct MPT verification puts Keccak-heavy work inside the user's PXE proving
path, which is especially painful for browser/mobile proving.

## Design Being Benchmarked

The design has two proving layers.

```text
Off-device prover / service
  fetches Ethereum proof data
  proves account/storage MPT relation in a standalone Noir circuit
  outputs UltraHonk proof + public inputs

User's Aztec private transaction
  verifies the UltraHonk proof via verify_honk_proof
  checks the public inputs match the expected state root/account/slot
  uses the proven storage value
```

The inner proof is not separately submitted to L1. It is verified inside the user's private Aztec
function. The user's private transaction proof is then included in the normal rollup proof flow.

This means the design inherits the normal private-context timing model. It can reduce user/PXE
proving cost, but it does not by itself solve same-block public access to the current L1 header or
current L1-derived facts.

## Existing Baseline

The repo already has a direct storage proof path:

- Noir contract: `noir-projects/noir-contracts/contracts/test/storage_proof_test_contract/src/main.nr`
- Fixture builder: `yarn-project/end-to-end/src/e2e_storage_proof/fixtures/storage_proof_fixture.ts`
- L1 proof fetcher: `yarn-project/end-to-end/src/e2e_storage_proof/fixtures/storage_proof_fetcher.ts`
- E2E test: `yarn-project/end-to-end/src/e2e_storage_proof/e2e_storage_proof.test.ts`
- Client benchmark: `yarn-project/end-to-end/src/bench/client_flows/storage_proof.test.ts`

That path verifies the MPT proof directly in Aztec execution. It is the main comparison target.

## What We Are Comparing

### Baseline A: Direct MPT Verification In User Transaction

The existing contract verifies:

```text
eth_state_root
  -> account proof for eth_address
  -> account.storage_hash
  -> storage proof for storage_slot_key
  -> storage_value
```

The user/PXE pays for the MPT path verification and associated Keccak work inside the private
transaction proving flow.

This benchmark represents the "do the work directly in the app circuit" approach.

### Candidate B: Offloaded MPT Proof With Recursive Verification

A standalone Noir circuit proves the same relation:

```text
eth_state_root
  -> account proof for eth_address
  -> account.storage_hash
  -> storage proof for storage_slot_key
  -> storage_value
```

It exposes public inputs:

```text
eth_state_root
eth_address
storage_slot_key
storage_value
```

Then an Aztec private function verifies:

```noir
verify_honk_proof(verification_key, proof, public_inputs, pinned_vk_hash);
```

and asserts:

```text
public_inputs.eth_state_root == expected_state_root
public_inputs.eth_address == expected_address
public_inputs.storage_slot_key == expected_slot
```

This benchmark represents the "proof-as-witness" offload approach.

## Why Mega/Goblin Matters

There are two different circuits involved:

| Layer | Role | Expected proving mode |
| --- | --- | --- |
| Inner circuit | Proves the MPT/storage fact | Standalone UltraHonk proof |
| Outer circuit | User's Aztec private function verifies the inner proof | Aztec private tx / ClientIVC / Mega-Goblin |

The recursive verifier must run in the outer Aztec private transaction circuit. This is the case
where recursive UltraHonk verification is expected to be cheap. The design prompt cites the key
distinction:

| Work inside outer circuit | Approximate cost |
| --- | ---: |
| Direct MPT/Keccak verification | hundreds of thousands of gates, roughly 278k+ for a storage proof shape |
| Recursive UltraHonk verification in Mega/Goblin | roughly 12k-15k gates |
| Recursive UltraHonk verification in plain Ultra | roughly 680k+ gates |

So the offload design is only attractive if the recursive verifier is measured in the Aztec private
transaction path, not in a plain Ultra circuit.

### Public Path Caveat

The plain Ultra recursive verifier is useful as a control case, but it is not the same thing as an
Aztec public function.

Aztec public functions execute in the AVM. They do not currently have the same app-level
`verify_honk_proof` path that private Noir circuits have. If AZIP-4 needs same-block public access to
verified L1 storage facts, that is a separate AVM/protocol design problem.

For this benchmark:

| Case | Meaning |
| --- | --- |
| Aztec private recursive verifier | The real app-level offload path being evaluated. |
| Plain Ultra recursive verifier | A control showing what happens outside the cheap Mega/Goblin path. |
| Public/AVM verifier | Out of scope unless the protocol adds a verifier opcode/precompile/path. |

## Benchmark Phases

### Phase 1: Direct Baseline

Use the existing storage proof benchmark unchanged.

Measure:

- PXE simulation time
- PXE proving time
- peak memory, if available from the benchmark harness or runtime profiling
- execution steps reported by the client benchmark harness
- transaction size / proof artifact size, if readily available
- success/failure under browser/WASM configuration, if practical

Output label:

```text
direct_storage_mpt
```

### Phase 2: Standalone Inner MPT Circuit

Create a standalone Noir package for the MPT relation. It should reuse as much of the existing
`storage_proof_test_contract/src/storage_proofs` logic as possible, but without Aztec contract
macros, capsules, or private/public calls.

Inputs:

```text
private:
  account proof nodes
  storage proof nodes
  account leaf data
  storage slot leaf data
  proof lengths

public:
  eth_state_root
  eth_address
  storage_slot_key
  storage_value
```

Measure:

- inner circuit gate count
- witness generation time
- UltraHonk proof generation time
- proof size
- verification key size
- public input count

Output label:

```text
inner_mpt_ultrahonk
```

This is service-side cost. It is not paid by the user's PXE in the offload design, but it is still
real work and must be measured.

### Phase 3: Recursive Verification Contract

Create an Aztec private contract function that accepts:

```text
verification_key
proof
public_inputs
```

It should:

1. Read or hardcode the expected VK hash.
2. Call `verify_honk_proof`.
3. Assert expected state root/account/slot.
4. Return or consume the proven storage value.

This can follow the structure of:

```text
docs/examples/contracts/recursive_verification_contract/src/main.nr
```

Measure:

- PXE simulation time
- PXE proving time
- peak memory, if available
- execution steps
- public/private input size added to the tx
- transaction size / proof artifact size, if readily available

Output label:

```text
recursive_storage_mpt_verify
```

This is the main user-side cost for the offload path.

### Phase 4: End-To-End Comparison

Compare:

```text
direct_storage_mpt
vs
recursive_storage_mpt_verify
```

Primary user-side question:

```text
PXE proving time/direct memory cost saved by recursive verification
```

Secondary system-wide question:

```text
inner_mpt_ultrahonk service cost + recursive_storage_mpt_verify user cost
vs
direct_storage_mpt user cost
```

The offload design can still be worthwhile even if total system work increases, because the goal is
to move heavy proving off constrained client devices and onto an optional proving service.

### Optional Phase 5: Plain Ultra Control

Build a standalone Noir outer verifier that only calls:

```noir
verify_honk_proof(verification_key, proof, public_inputs, pinned_vk_hash);
```

and prove it as a normal standalone circuit.

This is not the public Aztec path. It is a control measurement to confirm that recursive verification
is expensive outside the Aztec private/Mega-Goblin proving path.

Output label:

```text
plain_ultra_recursive_verify_control
```

## Header Binding

The first benchmark should isolate MPT/storage proof cost by using `eth_state_root` as the public
anchor.

AZIP-4 integration needs an additional binding step:

```text
canonical_l1_block_hash
  -> rlp_header
  -> state_root
  -> MPT proof
```

or, if a Poseidon-friendly header commitment exists:

```text
canonical_l1_header_commitment
  -> rlp_header
  -> state_root
  -> MPT proof
```

Do not include header binding in the first benchmark. Add it as a follow-up benchmark once the
storage-only offload path is understood.

Suggested follow-up labels:

```text
inner_header_plus_mpt_ultrahonk
recursive_header_plus_mpt_verify
```

## Receipt Proof Follow-Up

Receipt/log proofs should be treated as a later benchmark. They require separate parsing and proof
logic:

```text
rlp_header.receipts_root
  -> receipts trie proof
  -> receipt
  -> log/event fields
```

The storage benchmark should land first because the repo already has a direct storage proof baseline
and fixtures.

## Soundness Requirements

### Verification Key Pinning

The recursive verifier contract must not accept an arbitrary verification key as trusted.

Required check:

```text
hash(verification_key) == expected_vk_hash
```

The expected VK hash should be hardcoded for the benchmark or stored as a `PublicImmutable`, as in
the existing recursive verification example.

Without VK pinning, a prover could supply a proof for a different circuit that proves an irrelevant
statement.

### Public Input Binding

The outer Aztec function must check the verified public inputs. At minimum:

```text
state_root == expected_state_root
account == expected_account
slot == expected_slot
```

Later AZIP-4 integration should replace or supplement `expected_state_root` with a check against the
canonical L1 header hash or header commitment exposed by the protocol.

### Correctness Checks

The benchmark should verify correctness at each layer:

1. The direct baseline succeeds against the fixture.
2. The standalone inner MPT proof verifies natively with Barretenberg before being passed to Aztec.
3. The recursive verifier contract accepts the valid proof.
4. The recursive verifier contract rejects at least one tampered public input, such as a modified
   storage value or slot.
5. The recursive verifier contract rejects a proof/VK mismatch, or at minimum demonstrates that the
   pinned VK hash is checked.

These checks are required because the benchmark is only meaningful if the offloaded proof is proving
the same relation as the direct MPT baseline.

## Infrastructure We Have

Reusable today:

- Direct storage proof contract and benchmark.
- Fixture JSON format for account/storage proof nodes.
- L1 `eth_getProof` fetcher.
- Noir Keccak/MPT helper code under the storage proof test contract.
- Aztec private recursive verification example using `verify_honk_proof`.
- `bb_proof_verification` Noir library.
- Client-flow benchmark harness.

Missing for the benchmark:

- Standalone Noir MPT circuit.
- Script to generate recursive-friendly UltraHonk proof artifacts from `storage_proof.json`.
- Recursive storage proof verifier contract.
- Benchmark case comparing recursive verifier contract against the direct storage proof contract.

Missing for production:

- Metered proving service / RPC.
- Auth, payment, rate limiting, queueing, and caching.
- L1 archive-node access for proof generation.
- Privacy story for user queries.
- Governance or app-level process for pinning accepted VK hashes.

## L1 Data Source

The first benchmark can use the existing checked-in `storage_proof.json` fixture. It does not require
live access to an L1 node.

Live L1 access is needed when refreshing or expanding fixtures:

```text
eth_getProof
eth_getBlockByNumber / block header data
```

For production offload, the proving service needs archive-node access because it must fetch
historical account/storage proofs for the requested L1 block. For the initial benchmark, mocking via
fixtures is preferable because it keeps the measurement deterministic.

## Success Criteria

The benchmark supports the offload design if:

1. `recursive_storage_mpt_verify` has much lower user/PXE proving wall time than `direct_storage_mpt`.
2. `recursive_storage_mpt_verify` uses materially fewer client-side proving resources than
   `direct_storage_mpt`, especially memory.
3. Recursive verification remains in the expected Mega/Goblin cheap regime.
4. Proof and public input sizes are acceptable for Aztec transaction submission.
5. The standalone inner MPT proof can be generated reliably by server-class hardware.
6. VK pinning and public input binding are straightforward in the verifier contract.

The benchmark argues against the offload design if:

1. User/PXE proving wall time is not materially improved.
2. Client-side memory/resource usage is not materially improved.
3. Recursive verification falls into an expensive non-Mega path.
4. Proof artifacts are too large or awkward to pass into private functions.
5. Inner proof generation is too slow or memory-heavy for a practical proving service.
6. The implementation requires so much app-specific wiring that it is not reusable.

## Risks And Mitigations

### 1. Standalone Circuit Extraction

Risk:

The existing MPT logic lives inside `storage_proof_test_contract`. Some of it may depend on Aztec
contract types, serialization traits, capsules, or private/public call structure.

Mitigation:

- Reuse the pure helper modules first: `storage_proofs/types.nr`, `path_verification.nr`,
  `account_hash.nr`, `slot_hash.nr`, and the hash helpers.
- Keep the first standalone circuit narrow: one fixed fixture shape, account proof plus storage
  proof.
- Prefer small duplication over broad library refactors for the first benchmark.
- Refactor into a reusable library only after the benchmark path works.

### 2. Recursive-Friendly Proof Format

Risk:

The recursive verifier expects a proof generated for Noir recursive verification. A proof generated
for a native verifier or EVM verifier may be valid in isolation but fail inside `verify_honk_proof`.

Mitigation:

- Generate the inner proof with the recursive verifier target, equivalent to:

  ```text
  verifierTarget: "noir-recursive"
  ```

- Locally verify the proof with the same target before passing it into the Aztec contract.
- Add a small smoke test with a minimal known circuit before testing the full MPT circuit if proof
  formatting becomes unclear.

### 3. Capsules For Proof And VK

Risk:

The proof and VK are large enough that normal function arguments may be awkward, but capsules also
require exact serialization and key management.

Mitigation:

- Use capsules for the large proof and VK blobs in the intended benchmark path.
- Keep small expected values as normal function arguments.
- If capsules block early prototyping, temporarily pass proof/VK as arguments to validate recursive
  verification, then move them to capsules before collecting final benchmark numbers.
- Record the final transport choice in the benchmark output.

### 4. Public Input Encoding

Risk:

TypeScript, the standalone Noir circuit, and the Aztec verifier contract must agree exactly on field
layout. Mismatches in byte order, limb order, or value-length handling can make valid proofs fail or,
worse, make the benchmark measure a different statement than intended.

Mitigation:

- Define one public input struct and use it everywhere:

  ```noir
  pub struct StorageProofPublicInputs {
      pub state_root: [u64; 4],
      pub address: [u8; 20],
      pub slot_key: [u8; 32],
      pub value: [u8; 32],
      pub value_length: u8,
  }
  ```

- Match the existing fixture encoding:
  - Ethereum hashes/root values as `[u64; 4]` limbs using the existing fixture helper.
  - Address as `[u8; 20]`.
  - Slot key as `[u8; 32]`.
  - Slot value as `[u8; 32]` plus `value_length`.
- Add a TypeScript serialization test that produces the exact public input field array consumed by
  the recursive verifier.
- Add negative tests that tamper with `value`, `slot_key`, and `state_root`.

### 5. Memory Measurement

Risk:

Wall time and gate count are straightforward. Peak memory is harder to measure consistently,
especially through PXE, workers, or WASM.

Mitigation:

- Treat memory as best-effort in the first benchmark.
- Record native process RSS if available.
- Report whether the benchmark was native, Node/WASM, browser/WASM, or mobile-like.
- Do not block the first benchmark on precise browser/mobile memory capture.

### 6. Outer Verifier Cost Attribution

Risk:

The standalone inner circuit gate count is easy to report. The exact isolated gate cost of the
recursive verifier inside the Aztec private transaction path may not be exposed as a simple number by
the client-flow benchmark.

Mitigation:

- Use wall time from the client-flow benchmark as the primary user/PXE metric.
- Report execution steps and any available circuit/gate stats as secondary metrics.
- Compare the full direct path against the full recursive verifier path, not only isolated
  microbenchmarks.
- Keep the plain Ultra verifier only as a control, not as the public-path result.

### 7. VK Hash Pinning

Risk:

The verifier contract must pin the exact verification key hash expected by `verify_honk_proof`.
Using a hash from the wrong proof flavor or verifier target will fail verification.

Mitigation:

- Generate VK and proof with the same recursive target.
- Store the expected VK hash in `PublicImmutable` for the benchmark contract.
- Add a negative test with an incorrect VK hash.
- Document the exact command/API used to produce the proof and VK hash.

### 8. Benchmark Interpretation

Risk:

The offload path may reduce user/PXE proving cost while increasing total system work, because a
service still proves the MPT circuit and the user still verifies that proof recursively.

Mitigation:

- Report user-side and service-side costs separately.
- State clearly that the main goal is client/PXE UX and resource reduction.
- Do not present the result as a total-cost reduction unless the service-side measurements also show
  that.

## Open Questions

- Should the inner MPT circuit prove only storage inclusion, or both account and storage inclusion?
- Should `eth_state_root` be the first public input, or should the first version include full header
  binding to AZIP-4's canonical L1 block hash?
- Should the proof be passed as normal function arguments or via capsules?
- Should the VK be passed as an argument and checked against a stored hash, or should the contract
  hardcode a fixed VK?
- Can identical `(block, account, slot)` proof requests be cached and reused by many users?
- How much query privacy is lost if a remote service generates the proof?
- What is the expected pricing model for a production proving service?
- If public same-block access is required, what AVM/protocol primitive would replace or complement
  this private proof-offload path?

## Recommended First Milestone

Build the narrowest useful benchmark:

```text
storage_proof.json
  -> standalone MPT Noir circuit
  -> recursive-friendly UltraHonk proof
  -> Aztec private function verifies proof
  -> compare against existing direct storage proof benchmark
```

Do not include receipts, header binding, or a network proof service in the first milestone. Those are
important, but they should not obscure the core question: whether user-side MPT verification becomes
cheap enough when moved behind recursive UltraHonk verification.

## First Implementation Layout

The initial implementation uses the existing `storage_proof.json` fixture and adds three pieces:

| Piece | Path | Purpose |
|---|---|---|
| Standalone MPT circuit | `noir-projects/noir-protocol-circuits/crates/storage-proof-mpt` | Proves account + storage trie inclusion against a trusted Ethereum state root. |
| Recursive verifier contract | `noir-projects/noir-contracts/contracts/test/storage_proof_recursive_verifier_contract` | Private Aztec function loads VK/proof/public inputs from capsules, pins VK hash, verifies the UltraHonk proof, and checks expected state root/address/slot/value. |
| Recursive benchmark | `yarn-project/end-to-end/src/bench/client_flows/storage_proof_recursive.test.ts` | Profiles the user/PXE cost of recursive verification in the same client-flow harness as `storage_proof.test.ts`. |

The proof fixture generator is:

```text
yarn-project/end-to-end/src/e2e_storage_proof/fixtures/generate_mpt_ultrahonk_proof.ts
```

It must verify the generated proof locally before writing the fixture, and it should also mutate one
public input and confirm that verification fails. This confirms the proof is actually bound to the
state root/address/slot/value tuple that the Aztec verifier contract later checks.

Expected run order:

```bash
cd noir-projects/noir-protocol-circuits
yarn && yarn generate_variants
../../noir/noir-repo/target/release/nargo compile --package storage_proof_mpt --skip-brillig-constraints-check

cd ../../yarn-project/end-to-end
STORAGE_PROOF_MPT_ARTIFACT=../../noir-projects/noir-protocol-circuits/target/storage_proof_mpt.json \
  node --no-warnings --loader @swc-node/register/esm \
  src/e2e_storage_proof/fixtures/generate_mpt_ultrahonk_proof.ts
```

Then build noir contracts and generated TypeScript contract wrappers as usual, and run:

```bash
cd yarn-project/end-to-end
BENCHMARK_CONFIG=key_flows LOG_LEVEL=error yarn test:e2e src/bench/client_flows/storage_proof_recursive.test.ts
BENCHMARK_CONFIG=key_flows LOG_LEVEL=error yarn test:e2e src/bench/client_flows/storage_proof.test.ts
```

The recursive benchmark should not be promoted to the regular benchmark queue until the recursive
proof fixture is generated reproducibly in the normal build flow.

## Initial Local Results

Measured locally on 2026-06-26 with:

```bash
cd yarn-project/end-to-end
BENCHMARK_CONFIG=key_flows LOG_LEVEL=error BENCH_OUTPUT=bench-out \
  yarn test:e2e src/bench/client_flows/storage_proof_recursive.test.ts --runInBand
BENCHMARK_CONFIG=key_flows LOG_LEVEL=error BENCH_OUTPUT=bench-out \
  yarn test:e2e src/bench/client_flows/storage_proof.test.ts --runInBand
```

User/PXE-side benchmark results:

| Flow | Total | Witgen | Gate count | RPC calls |
|---|---:|---:|---:|---:|
| Recursive UltraHonk verify | 771.30 ms | 459.34 ms | 380,768 | 8 |
| Direct 7-layer storage proof | 1,536.28 ms | 1,209.07 ms | 1,798,315 | 5 |

In this run, the recursive path was about 2.0x faster end-to-end, 2.6x faster in witgen, and 4.7x
smaller by total gate count. The recursive path used three extra RPC calls in this harness.

Standalone inner proof generation is a separate service-side cost. The generated fixture measured:

| Step | Time |
|---|---:|
| Witness generation | 1,405.86 ms |
| UltraHonk proving | 67,884.54 ms |
| Native proof verification | 18,285.87 ms |

The recursive proof fixture contained 115 VK fields, 458 proof fields, and 89 public inputs. These
numbers should be treated as local first-pass measurements, not stable CI benchmarks.
