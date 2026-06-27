# AZIP-4 Storage Proof Offload Benchmark

This branch prototypes and benchmarks an UltraHonk proof-offload path for Ethereum account/storage
MPT proofs.

The question being tested is narrow:

> Is it cheaper for an Aztec private function to recursively verify a standalone UltraHonk proof of
> an MPT storage proof than to open the keccak/MPT proof directly inside the user's private circuit?

The first local answer is yes for the user/PXE side of the flow. It is not yet evidence that the
full production system is cheaper end-to-end, because the standalone MPT proof still has to be
generated somewhere.

All paths and commands below are repo-root-relative unless stated otherwise.

## What This Branch Adds

| Piece | Path | Purpose |
|---|---|---|
| Standalone MPT circuit | `noir-projects/noir-protocol-circuits/crates/storage-proof-mpt` | Proves account + storage trie inclusion against a trusted Ethereum state root. |
| Recursive verifier contract | `noir-projects/noir-contracts/contracts/test/storage_proof_recursive_verifier_contract` | Private function loads VK/proof/public inputs from capsules, pins the VK hash, recursively verifies the UltraHonk proof, and checks the expected state root/address/slot/value. |
| Proof fixture generator | `yarn-project/end-to-end/src/e2e_storage_proof/fixtures/generate_mpt_ultrahonk_proof.ts` | Builds the standalone proof fixture from `storage_proof.json`, verifies it natively, and writes recursive verifier inputs. |
| Recursive benchmark | `yarn-project/end-to-end/src/bench/client_flows/storage_proof_recursive.test.ts` | Compares the recursive verifier path against the existing direct storage proof benchmark harness. |
| Benchmark notes | `proof-storage-azip-4.md` | Working notes and benchmark plan. |

The recursive proof fixture is written to:

```text
yarn-project/end-to-end/src/e2e_storage_proof/fixtures/storage_proof_ultrahonk.json
```

## How The Benchmark Works

The direct baseline is the existing storage proof path:

```text
private Aztec function
  -> opens account MPT proof
  -> opens storage MPT proof
  -> pays keccak/MPT cost inside the user's private circuit
```

The recursive path is:

```text
storage_proof.json fixture
  -> standalone Noir MPT circuit
  -> standalone UltraHonk proof
  -> proof/VK/public inputs loaded into an Aztec private function via capsules
  -> private function calls verify_honk_proof
  -> private function asserts public inputs match expected state root/address/slot/value
```

The benchmark includes the Aztec private call, PXE simulation/profile path, account entrypoint,
sponsored FPC payment, capsule loading, and recursive verifier contract. It does not include a live
proof service, live L1 node access, or fetching `eth_getProof`.

## Proof Setup

Yes, this branch adds a custom standalone Noir circuit:

```text
noir-projects/noir-protocol-circuits/crates/storage-proof-mpt
```

That circuit proves one Ethereum storage slot opening. It does not prove all storage for an account,
and it does not prove receipt inclusion or block-header validity.

The circuit has private inputs for the witness data:

| Private input | Meaning |
|---|---|
| `account` | Decoded Ethereum account fields: nonce, balance, address, storage root, code hash. |
| `account_nodes` | MPT nodes proving the account exists under the Ethereum state root. |
| `account_node_length` | Number of real account proof nodes in the padded array. |
| `storage_nodes` | MPT nodes proving the storage slot exists under the account storage root. |
| `storage_node_length` | Number of real storage proof nodes in the padded array. |

It exposes these public inputs:

| Public input | Meaning |
|---|---|
| `state_root` | Trusted Ethereum state root, encoded as four `u64` limbs. |
| `address` | Ethereum account address being opened. |
| `slot_key` | The 32-byte storage slot key. |
| `value` | The 32-byte storage slot value. |
| `value_length` | RLP/value length metadata for the slot value. |

The proven statement is:

```text
Given public (state_root, address, slot_key, value, value_length),
I know account data, account MPT nodes, and storage MPT nodes such that:

1. keccak(address) leads through account_nodes from state_root to an account leaf;
2. that account leaf hashes to the supplied account data;
3. the account data contains a storage root;
4. keccak(slot_key) leads through storage_nodes from that storage root to a slot leaf;
5. that slot leaf hashes to (value, value_length).
```

The fixture generator then turns this circuit execution into a standalone UltraHonk proof. The Aztec
private verifier contract does not re-run the MPT logic. It loads the proof, VK, and public inputs
from capsules, checks the VK hash against a constructor-pinned value, calls `verify_honk_proof`, and
then checks that the verified public inputs equal the function arguments.

## Initial Local Results

Measured locally on 2026-06-26 with:

```text
CPU: Intel(R) Core(TM) i9-14900K
Logical CPUs: 32
Platform: linux x64
RAM: ~62.5 GB
```

| Flow | Total | Witgen | Gate count | RPC calls |
|---|---:|---:|---:|---:|
| Recursive UltraHonk verify | 771.30 ms | 459.34 ms | 380,768 | 8 |
| Direct 7-layer storage proof | 1,536.28 ms | 1,209.07 ms | 1,798,315 | 5 |

In this run, the recursive path was about:

- 2.0x faster end-to-end
- 2.6x faster in witgen
- 4.7x smaller by total gate count

The recursive path used three more RPC calls in this harness.

## Service-Side Cost

The standalone MPT proof is not free. It is generated before the recursive benchmark and treated as
a fixture. Local fixture-generation timings were:

| Step | Time |
|---|---:|
| Witness generation | 1,405.86 ms |
| UltraHonk proving | 67,884.54 ms |
| Native proof verification | 18,285.87 ms |

The native verification step is only a fixture-generation sanity check. It is not paid by the
Aztec user/PXE benchmark. In production, a user would not need to trust a service-side native
verification because the user's private circuit recursively verifies the proof again.

The generated recursive proof fixture contained:

| Item | Count |
|---|---:|
| VK fields | 115 |
| Proof fields | 458 |
| Public inputs | 89 |

## Correctness Checks Present Today

The prototype currently checks correctness in three places:

1. The standalone MPT circuit proves the account proof and storage proof against a public
   `state_root/address/slot/value/value_length` tuple.
2. The fixture generator runs native Barretenberg verification before writing the fixture.
3. The fixture generator mutates one public input and asserts that native verification fails.
4. The Aztec private verifier contract pins the VK hash, calls `verify_honk_proof`, and then checks
   that the verified public inputs match the expected function arguments.

This means a malicious proof provider should not be able to substitute an arbitrary proof unless it
also matches the pinned VK and public inputs.

## Important Limitations

These numbers should be read as a first-pass local benchmark, not as final product economics.

- The fixture is static. There is no live L1 node, archive node, or `eth_getProof` request in the
  benchmark.
- The benchmark starts from a trusted Ethereum state root. It does not yet bind that state root to
  the AZIP-4 L1 block hash commitment.
- The proof service does not exist. The proof is generated locally ahead of time.
- The service-side proving cost is large in this local run. The user-side win only makes sense if
  proofs are generated off-device, cached, reused, or otherwise made available without the user's
  PXE proving them.
- The path uses a private function, so it measures the Mega/Goblin recursive-verification regime.
  It does not answer whether public AVM same-block access is cheap.
- Public input layout is currently fixed manually at 89 fields.
- The benchmark covers storage MPT proof verification only. Receipts, richer header binding, and
  generalized proof requests are not included.
- More negative tests should be added for wrong VK hash, tampered proof capsule, tampered public
  input capsule, and mismatched expected function arguments.

## How To Run

The condensed reproduction flow is:

```bash
cd noir-projects/noir-protocol-circuits
yarn && yarn generate_variants
../../noir/noir-repo/target/release/nargo compile --package storage_proof_mpt --skip-brillig-constraints-check

cd ../../yarn-project/end-to-end
STORAGE_PROOF_MPT_ARTIFACT=../../noir-projects/noir-protocol-circuits/target/storage_proof_mpt.json \
  node --no-warnings --loader @swc-node/register/esm \
  src/e2e_storage_proof/fixtures/generate_mpt_ultrahonk_proof.ts
```

Build the recursive verifier contract and generated TypeScript wrappers as needed, then run the
recursive benchmark and the existing direct storage proof benchmark from `yarn-project/end-to-end`.
Benchmark JSON is emitted under `yarn-project/end-to-end/bench-out/`.
