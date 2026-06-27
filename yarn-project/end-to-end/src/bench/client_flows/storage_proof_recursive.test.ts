import { AztecAddress } from '@aztec/aztec.js/addresses';
import type { ContractInstanceWithAddress, SimulateInteractionOptions } from '@aztec/aztec.js/contracts';
import { FPCContract } from '@aztec/noir-contracts.js/FPC';
import { SponsoredFPCContract } from '@aztec/noir-contracts.js/SponsoredFPC';
import { TokenContract } from '@aztec/noir-contracts.js/Token';
import { StorageProofRecursiveVerifierContract } from '@aztec/noir-test-contracts.js/StorageProofRecursiveVerifier';

import { jest } from '@jest/globals';

import {
  buildRecursiveStorageProofCapsules,
  loadRecursiveStorageProofArgs,
} from '../../e2e_storage_proof/fixtures/storage_proof_fixture.js';
import type { TestWallet } from '../../test-wallet/test_wallet.js';
import { captureProfile, expectedExecutionSteps } from './benchmark.js';
import { type AccountType, type BenchmarkingFeePaymentMethod, ClientFlowsBenchmark } from './client_flows_benchmark.js';

jest.setTimeout(300_000);

describe('Recursive storage proof benchmark', () => {
  const t = new ClientFlowsBenchmark('storage_proof_recursive');
  let userWallet: TestWallet;
  let adminAddress: AztecAddress;
  let bananaFPCInstance: ContractInstanceWithAddress;
  let bananaCoinInstance: ContractInstanceWithAddress;
  let sponsoredFPCInstance: ContractInstanceWithAddress;
  let recursiveVerifierContract: StorageProofRecursiveVerifierContract;
  let recursiveVerifierInstance: ContractInstanceWithAddress;
  const config = t.config.storageProof;

  beforeAll(async () => {
    await t.setup();
    await t.applyDeployBananaToken();
    await t.applyFPCSetup();
    await t.applyDeploySponsoredFPC();

    const { vkHash } = loadRecursiveStorageProofArgs();
    const deployed = await StorageProofRecursiveVerifierContract.deploy(t.adminWallet, vkHash).send({
      from: t.adminAddress,
    });
    recursiveVerifierContract = deployed.contract;
    recursiveVerifierInstance = deployed.instance;

    ({ userWallet, adminAddress, bananaFPCInstance, bananaCoinInstance, sponsoredFPCInstance } = t);
  });

  afterAll(async () => {
    await t.teardown();
  });

  for (const accountType of config.accounts) {
    recursiveStorageProofBenchmark(accountType);
  }

  function recursiveStorageProofBenchmark(accountType: AccountType) {
    return describe(`Recursive storage proof benchmark for ${accountType}`, () => {
      let benchysAddress: AztecAddress;

      beforeAll(async () => {
        benchysAddress = await t.createAndFundBenchmarkingAccountOnUserWallet(accountType);
        await t.mintPrivateBananas(1000n * 10n ** 18n, benchysAddress);
        await userWallet.registerSender(adminAddress);
        await userWallet.registerContract(bananaFPCInstance, FPCContract.artifact);
        await userWallet.registerContract(bananaCoinInstance, TokenContract.artifact);
        await userWallet.registerContract(sponsoredFPCInstance, SponsoredFPCContract.artifact);
        await userWallet.registerContract(recursiveVerifierInstance, StorageProofRecursiveVerifierContract.artifact);
      });

      for (const paymentMethod of config.feePaymentMethods) {
        recursiveStorageProofTest(paymentMethod);
      }

      function recursiveStorageProofTest(benchmarkingPaymentMethod: BenchmarkingFeePaymentMethod) {
        return it(`${accountType} recursive storage proof pays using ${benchmarkingPaymentMethod}`, async () => {
          const paymentMethod = t.paymentMethods[benchmarkingPaymentMethod];
          const { ethAddress, slotKey, slotContents, root } = loadRecursiveStorageProofArgs();
          const contract = StorageProofRecursiveVerifierContract.at(recursiveVerifierContract.address, userWallet);
          const capsules = buildRecursiveStorageProofCapsules(contract.address);

          const interaction = contract.methods
            .verify_storage_proof(root, ethAddress, slotKey, slotContents.value, slotContents.value_length)
            .with({ capsules });

          const options: SimulateInteractionOptions = {
            from: benchysAddress,
            fee: { paymentMethod: await paymentMethod.forWallet(userWallet, benchysAddress) },
          };

          await captureProfile(
            `${accountType}+storage_proof_recursive_verify+${benchmarkingPaymentMethod}`,
            interaction,
            options,
            expectedExecutionSteps(
              1 + // Account entrypoint
                paymentMethod.apps + // Payment method apps
                1, // Recursive storage proof verifier entry
            ),
          );

          if (process.env.SANITY_CHECKS) {
            const { receipt: tx } = await interaction.send(options);
            expect(tx.transactionFee!).toBeGreaterThan(0n);
            expect(tx.hasExecutionSucceeded()).toBe(true);
          }
        });
      }
    });
  }
});
