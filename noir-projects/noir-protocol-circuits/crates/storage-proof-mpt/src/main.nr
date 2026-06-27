mod storage_proofs;

use crate::storage_proofs::{
    account_hash::compute_account_hash,
    keccak::keccak256,
    path_verification::verify_path_section,
    slot_hash::compute_slot_hash,
    types::{Account, Node, StorageSlot},
    utils::to_nibbles,
};

global MAX_ACCOUNT_PROOF_LENGTH: u32 = 15;
global MAX_STORAGE_PROOF_LENGTH: u32 = 10;

struct StorageProofPublicInputs {
    state_root: [u64; 4],
    address: [u8; 20],
    slot_key: [u8; 32],
    value: [u8; 32],
    value_length: u8,
}

fn main(
    account: Account,
    account_nodes: [Node; MAX_ACCOUNT_PROOF_LENGTH],
    account_node_length: u32,
    storage_nodes: [Node; MAX_STORAGE_PROOF_LENGTH],
    storage_node_length: u32,
    public_inputs: pub StorageProofPublicInputs,
) {
    assert(account.address == public_inputs.address);

    let account_trie_key = keccak256(public_inputs.address, 20);
    let (account_leaf_hash, account_nibble_index) = verify_path_section(
        account_nodes,
        account_node_length,
        0,
        public_inputs.state_root,
        to_nibbles(account_trie_key),
    );

    assert(
        account_leaf_hash == compute_account_hash(account, account_nibble_index, account_trie_key),
    );

    let storage_trie_key = keccak256(public_inputs.slot_key, 32);
    let (slot_leaf_hash, slot_nibble_index) = verify_path_section(
        storage_nodes,
        storage_node_length,
        0,
        account.storage_hash,
        to_nibbles(storage_trie_key),
    );

    let slot = StorageSlot { value: public_inputs.value, value_length: public_inputs.value_length };
    assert(slot_leaf_hash == compute_slot_hash(slot, slot_nibble_index, storage_trie_key));
}
