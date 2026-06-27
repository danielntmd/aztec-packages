use aztec::macros::aztec;

#[aztec]
pub contract StorageProofRecursiveVerifier {
    use aztec::{
        macros::{functions::{external, initializer, internal}, storage::storage},
        oracle::capsules,
        protocol::{address::{AztecAddress, EthAddress}, traits::ToField},
        state_vars::PublicImmutable,
    };
    use bb_proof_verification::{UltraHonkVerificationKey, UltraHonkZKProof, verify_honk_proof};

    global PUBLIC_INPUTS_LENGTH: u32 = 89;

    global VK_CAPSULE_SLOT: Field = 200;
    global PROOF_CAPSULE_SLOT: Field = 201;
    global PUBLIC_INPUTS_CAPSULE_SLOT: Field = 202;

    #[storage]
    struct Storage<Context> {
        vk_hash: PublicImmutable<Field, Context>,
    }

    #[initializer]
    #[external("public")]
    fn constructor(vk_hash: Field) {
        self.storage.vk_hash.initialize(vk_hash);
    }

    #[external("private")]
    fn verify_storage_proof(
        expected_state_root: [u64; 4],
        expected_address: EthAddress,
        expected_slot_key: [u8; 32],
        expected_value: [u8; 32],
        expected_value_length: u8,
    ) {
        // Safety: The recursive verifier constrains the VK against the pinned VK hash.
        let verification_key: UltraHonkVerificationKey =
            unsafe { capsules::load(self.address, VK_CAPSULE_SLOT, AztecAddress::zero()).unwrap() };
        // Safety: The recursive verifier constrains the proof against the pinned VK and public inputs.
        let proof: UltraHonkZKProof =
            unsafe { capsules::load(self.address, PROOF_CAPSULE_SLOT, AztecAddress::zero()).unwrap() };
        // Safety: The public inputs are checked against the expected function arguments below.
        let public_inputs: [Field; PUBLIC_INPUTS_LENGTH] = unsafe {
            capsules::load(
                self.address,
                PUBLIC_INPUTS_CAPSULE_SLOT,
                AztecAddress::zero(),
            )
                .unwrap()
        };

        let vk_hash = self.storage.vk_hash.read();
        verify_honk_proof(verification_key, proof, public_inputs, vk_hash);

        self.internal.assert_public_inputs_match(
            public_inputs,
            expected_state_root,
            expected_address,
            expected_slot_key,
            expected_value,
            expected_value_length,
        );
    }

    #[internal("private")]
    fn assert_public_inputs_match(
        public_inputs: [Field; PUBLIC_INPUTS_LENGTH],
        expected_state_root: [u64; 4],
        expected_address: EthAddress,
        expected_slot_key: [u8; 32],
        expected_value: [u8; 32],
        expected_value_length: u8,
    ) {
        for i in 0..4 {
            assert_eq(public_inputs[i], expected_state_root[i] as Field);
        }

        let expected_address_bytes = expected_address.to_field().to_be_bytes::<20>();
        for i in 0..20 {
            assert_eq(public_inputs[4 + i], expected_address_bytes[i] as Field);
        }

        for i in 0..32 {
            assert_eq(public_inputs[24 + i], expected_slot_key[i] as Field);
            assert_eq(public_inputs[56 + i], expected_value[i] as Field);
        }

        assert_eq(public_inputs[88], expected_value_length as Field);
    }
}
