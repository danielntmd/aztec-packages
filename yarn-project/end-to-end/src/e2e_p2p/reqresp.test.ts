import type { AztecNodeService } from '@aztec/aztec-node';
import { SentTx, Tx, createLogger, sleep } from '@aztec/aztec.js';
import { RollupContract } from '@aztec/ethereum';
import { timesAsync } from '@aztec/foundation/collection';

import { jest } from '@jest/globals';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { shouldCollectMetrics } from '../fixtures/fixtures.js';
import { createNodes } from '../fixtures/setup_p2p_test.js';
import { P2PNetworkTest, SHORTENED_BLOCK_TIME_CONFIG_NO_PRUNES, WAIT_FOR_TX_TIMEOUT } from './p2p_network.js';
import { createPXEServiceAndPrepareTransactions } from './shared.js';

// Don't set this to a higher value than 9 because each node will use a different L1 publisher account and anvil seeds
const NUM_VALIDATORS = 6;
const NUM_TXS_PER_NODE = 2;
const BOOT_NODE_UDP_PORT = 4500;

const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'reqresp-'));

describe('e2e_p2p_reqresp_tx', () => {
  let t: P2PNetworkTest;
  let nodes: AztecNodeService[];

  beforeEach(async () => {
    const beforeEachStart = Date.now();

    const networkTestCreateStart = Date.now();
    t = await P2PNetworkTest.create({
      testName: 'e2e_p2p_reqresp_tx',
      numberOfNodes: 0,
      numberOfValidators: NUM_VALIDATORS,
      basePort: BOOT_NODE_UDP_PORT,
      // To collect metrics - run in aztec-packages `docker compose --profile metrics up`
      metricsPort: shouldCollectMetrics(),
      initialConfig: {
        ...SHORTENED_BLOCK_TIME_CONFIG_NO_PRUNES,
        listenAddress: '127.0.0.1',
        aztecEpochDuration: 64, // stable committee
      },
    });
    const networkTestCreateEnd = Date.now();

    // Use a debug logger that will be available after t is created
    const logger = createLogger('reqresp_test_debug');
    logger.info(`[REQRESP_DEBUG] beforeEach started at ${new Date().toISOString()}`);
    logger.info(
      `[REQRESP_DEBUG] P2PNetworkTest.create completed in ${networkTestCreateEnd - networkTestCreateStart}ms`,
    );

    t.logger.info('[REQRESP_DEBUG] Testing Changes....');

    const snapshotsStart = Date.now();
    await t.applyBaseSnapshots();
    const snapshotsEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] applyBaseSnapshots completed in ${snapshotsEnd - snapshotsStart}ms`);

    const setupStart = Date.now();
    await t.setup();
    const setupEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] setup completed in ${setupEnd - setupStart}ms`);

    const beforeEachEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] beforeEach completed in ${beforeEachEnd - beforeEachStart}ms`);
  });

  afterEach(async () => {
    const afterEachStart = Date.now();
    const logger = createLogger('reqresp_test_debug');
    logger.info(`[REQRESP_DEBUG] afterEach started at ${new Date().toISOString()}`);

    const stopNodesStart = Date.now();
    await t.stopNodes(nodes);
    const stopNodesEnd = Date.now();
    logger.info(`[REQRESP_DEBUG] stopNodes completed in ${stopNodesEnd - stopNodesStart}ms`);

    const teardownStart = Date.now();
    await t.teardown();
    const teardownEnd = Date.now();
    logger.info(`[REQRESP_DEBUG] teardown completed in ${teardownEnd - teardownStart}ms`);

    const cleanupStart = Date.now();
    for (let i = 0; i < NUM_VALIDATORS; i++) {
      fs.rmSync(`${DATA_DIR}-${i}`, { recursive: true, force: true, maxRetries: 3 });
    }
    const cleanupEnd = Date.now();
    logger.info(`[REQRESP_DEBUG] cleanup completed in ${cleanupEnd - cleanupStart}ms`);

    const afterEachEnd = Date.now();
    logger.info(`[REQRESP_DEBUG] afterEach completed in ${afterEachEnd - afterEachStart}ms`);
  });

  const getNodePort = (nodeIndex: number) => BOOT_NODE_UDP_PORT + 1 + nodeIndex;

  it('should produce an attestation by requesting tx data over the p2p network', async () => {
    /**
     * Birds eye overview of the test
     * 1. We spin up x nodes
     * 2. We turn off receiving a tx via gossip from two of the nodes
     * 3. We send a transactions and gossip it to other nodes
     * 4. The disabled nodes will receive an attestation that it does not have the data for
     * 5. They will request this data over the p2p layer
     * 6. We receive all of the attestations that we need and we produce the block
     *
     * Note: we do not attempt to let this node produce a block, as it will not have received any transactions
     *       from the other pxes.
     */

    const testStartTime = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Test started at ${new Date().toISOString()}`);

    if (!t.bootstrapNodeEnr) {
      throw new Error('Bootstrap node ENR is not available');
    }

    const nodeCreationStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Creating ${NUM_VALIDATORS} nodes at ${new Date().toISOString()}`);
    nodes = await createNodes(
      t.ctx.aztecNodeConfig,
      t.ctx.dateProvider,
      t.bootstrapNodeEnr,
      NUM_VALIDATORS,
      BOOT_NODE_UDP_PORT,
      t.prefilledPublicData,
      DATA_DIR,
      shouldCollectMetrics(),
    );
    const nodeCreationEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Node creation completed in ${nodeCreationEnd - nodeCreationStart}ms`);

    const connectionWaitStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Sleeping 4000ms to allow nodes to connect at ${new Date().toISOString()}`);
    await sleep(4000);
    const connectionWaitEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Connection wait completed in ${connectionWaitEnd - connectionWaitStart}ms`);

    // Log peer connections for each node
    for (let i = 0; i < nodes.length; i++) {
      const peers = (nodes[i] as any).p2pClient?.p2pService?.getPeers();
      t.logger.info(`[REQRESP_DEBUG] Node ${i} (port ${getNodePort(i)}) has ${peers?.length || 0} connected peers`);
    }

    const accountSetupStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Setting up account at ${new Date().toISOString()}`);
    await t.setupAccount();
    const accountSetupEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Account setup completed in ${accountSetupEnd - accountSetupStart}ms`);

    const txPrepStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Preparing transactions to send at ${new Date().toISOString()}`);
    const contexts = await timesAsync(2, () =>
      createPXEServiceAndPrepareTransactions(t.logger, t.ctx.aztecNode, NUM_TXS_PER_NODE, t.fundedAccount),
    );
    const txPrepEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Transaction preparation completed in ${txPrepEnd - txPrepStart}ms`);
    t.logger.info(
      `[REQRESP_DEBUG] Prepared ${contexts.length} contexts with ${contexts.reduce((sum, ctx) => sum + ctx.txs.length, 0)} total transactions`,
    );

    const nodeRemovalStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Removing initial node at ${new Date().toISOString()}`);
    await t.removeInitialNode();
    const nodeRemovalEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Initial node removal completed in ${nodeRemovalEnd - nodeRemovalStart}ms`);

    const slotAdvanceStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Starting fresh slot at ${new Date().toISOString()}`);
    const [timestamp] = await t.ctx.cheatCodes.rollup.advanceToNextSlot();
    t.ctx.dateProvider.setTime(Number(timestamp) * 1000);
    const slotAdvanceEnd = Date.now();
    t.logger.info(
      `[REQRESP_DEBUG] Slot advance completed in ${slotAdvanceEnd - slotAdvanceStart}ms, new timestamp: ${timestamp}`,
    );

    const proposerSelectionStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Getting proposer indexes at ${new Date().toISOString()}`);
    const { proposerIndexes, nodesToTurnOffTxGossip } = await getProposerIndexes();
    const proposerSelectionEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Proposer selection completed in ${proposerSelectionEnd - proposerSelectionStart}ms`);
    t.logger.info(
      `[REQRESP_DEBUG] Turning off tx gossip for nodes: ${nodesToTurnOffTxGossip.map(getNodePort)} (${nodesToTurnOffTxGossip.length} nodes)`,
    );
    t.logger.info(
      `[REQRESP_DEBUG] Sending txs to proposer nodes: ${proposerIndexes.map(getNodePort)} (${proposerIndexes.length} nodes)`,
    );

    // Replace the p2p node implementation of some of the nodes with a spy such that it does not store transactions that are gossiped to it
    // Original implementation of `handleGossipedTx` will store received transactions in the tx pool.
    // We chose the first 2 nodes that will be the proposers for the next few slots
    const gossipDisableStart = Date.now();
    t.logger.info(
      `[REQRESP_DEBUG] Disabling gossip for ${nodesToTurnOffTxGossip.length} nodes at ${new Date().toISOString()}`,
    );
    for (const nodeIndex of nodesToTurnOffTxGossip) {
      const logger = createLogger(`p2p:${getNodePort(nodeIndex)}`);
      t.logger.info(`[REQRESP_DEBUG] Disabling gossip for node ${nodeIndex} (port ${getNodePort(nodeIndex)})`);
      jest.spyOn((nodes[nodeIndex] as any).p2pClient.p2pService, 'handleGossipedTx').mockImplementation(((
        payloadData: Buffer,
      ) => {
        const txHash = Tx.fromBuffer(payloadData).getTxHash();
        logger.info(
          `[REQRESP_DEBUG] Skipping storage of gossiped transaction ${txHash.toString()} on node ${nodeIndex}`,
        );
        return Promise.resolve();
      }) as any);
    }
    const gossipDisableEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Gossip disabling completed in ${gossipDisableEnd - gossipDisableStart}ms`);

    // We send the tx to the proposer nodes directly, ignoring the pxe and node in each context
    // We cannot just call tx.send since they were created using a pxe wired to the first node which is now stopped
    const txSendStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Sending transactions through proposer nodes at ${new Date().toISOString()}`);
    const sentTxs = contexts.map((c, i) =>
      c.txs.map((tx, txIndex) => {
        const node = nodes[proposerIndexes[i]];
        const txHash = tx.getTxHash().toString();
        t.logger.info(
          `[REQRESP_DEBUG] Sending tx ${i}-${txIndex} (${txHash}) to proposer node ${proposerIndexes[i]} (port ${getNodePort(proposerIndexes[i])})`,
        );
        void node.sendTx(tx).catch(err => t.logger.error(`[REQRESP_DEBUG] Error sending tx ${txHash}: ${err}`));
        return new SentTx(node, () => Promise.resolve(tx.getTxHash()));
      }),
    );
    const txSendEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] Transaction sending completed in ${txSendEnd - txSendStart}ms`);
    t.logger.info(`[REQRESP_DEBUG] Sent ${sentTxs.reduce((sum, txs) => sum + txs.length, 0)} total transactions`);

    const txWaitStart = Date.now();
    const timeoutMs = WAIT_FOR_TX_TIMEOUT * 1.5;
    t.logger.info(`[REQRESP_DEBUG] Waiting for all transactions to be mined at ${new Date().toISOString()}`);
    t.logger.info(`[REQRESP_DEBUG] Using timeout of ${timeoutMs}ms (WAIT_FOR_TX_TIMEOUT * 1.5)`);
    t.logger.info(
      `[REQRESP_DEBUG] Total transactions to wait for: ${sentTxs.reduce((sum, txs) => sum + txs.length, 0)}`,
    );

    await Promise.all(
      sentTxs.flatMap((txs, i) =>
        txs.map(async (tx, j) => {
          const txWaitStartIndividual = Date.now();
          const txHash = (await tx.getTxHash()).toString();
          t.logger.info(
            `[REQRESP_DEBUG] Waiting for tx ${i}-${j} (${txHash}) to be mined at ${new Date().toISOString()}`,
          );

          try {
            await tx.wait({ timeout: timeoutMs }); // more transactions in this test so allow more time
            const txWaitEndIndividual = Date.now();
            t.logger.info(
              `[REQRESP_DEBUG] Tx ${i}-${j} (${txHash}) has been mined in ${txWaitEndIndividual - txWaitStartIndividual}ms`,
            );
          } catch (error) {
            const txWaitEndIndividual = Date.now();
            t.logger.error(
              `[REQRESP_DEBUG] Tx ${i}-${j} (${txHash}) failed to mine after ${txWaitEndIndividual - txWaitStartIndividual}ms: ${error}`,
            );
            throw error;
          }
        }),
      ),
    );

    const txWaitEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] All transactions mined in ${txWaitEnd - txWaitStart}ms`);
    const totalTestTime = txWaitEnd - testStartTime;
    t.logger.info(`[REQRESP_DEBUG] Total test time so far: ${totalTestTime}ms`);
  });

  /**
   * Get the indexes in the nodes array that will produce the next few blocks
   */
  async function getProposerIndexes() {
    const fnStart = Date.now();
    t.logger.info(`[REQRESP_DEBUG] getProposerIndexes started at ${new Date().toISOString()}`);

    // Get the nodes for the next set of slots
    const contractInitStart = Date.now();
    const rollupContract = new RollupContract(
      t.ctx.deployL1ContractsValues.l1Client,
      t.ctx.deployL1ContractsValues.l1ContractAddresses.rollupAddress,
    );
    const contractInitEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] RollupContract initialization took ${contractInitEnd - contractInitStart}ms`);

    const attestersStart = Date.now();
    const attesters = await rollupContract.getAttesters();
    const attestersEnd = Date.now();
    t.logger.info(
      `[REQRESP_DEBUG] getAttesters() took ${attestersEnd - attestersStart}ms, got ${attesters.length} attesters`,
    );

    const timestampStart = Date.now();
    const currentTime = await t.ctx.cheatCodes.eth.timestamp();
    const timestampEnd = Date.now();
    t.logger.info(
      `[REQRESP_DEBUG] Getting timestamp took ${timestampEnd - timestampStart}ms, currentTime: ${currentTime}`,
    );

    const slotDurationStart = Date.now();
    const slotDuration = await rollupContract.getSlotDuration();
    const slotDurationEnd = Date.now();
    t.logger.info(
      `[REQRESP_DEBUG] getSlotDuration() took ${slotDurationEnd - slotDurationStart}ms, slotDuration: ${slotDuration}`,
    );

    const proposers = [];
    const proposerQueryStart = Date.now();

    for (let i = 0; i < 3; i++) {
      const nextSlot = BigInt(currentTime) + BigInt(i) * BigInt(slotDuration);
      const proposerStart = Date.now();
      const proposer = await rollupContract.getProposerAt(nextSlot);
      const proposerEnd = Date.now();
      t.logger.info(
        `[REQRESP_DEBUG] getProposerAt(${nextSlot}) took ${proposerEnd - proposerStart}ms, proposer: ${proposer}`,
      );
      proposers.push(proposer);
    }
    const proposerQueryEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] All proposer queries took ${proposerQueryEnd - proposerQueryStart}ms`);

    // Get the indexes of the nodes that are responsible for the next two slots
    const proposerIndexes = proposers.map(proposer => attesters.indexOf(proposer as `0x${string}`));
    t.logger.info(`[REQRESP_DEBUG] Proposer indexes: ${proposerIndexes.join(', ')}`);

    if (proposerIndexes.some(i => i === -1)) {
      const errorMsg = `Proposer index not found for proposer (proposers=${proposers.join(',')}, indices=${proposerIndexes.join(',')})`;
      t.logger.error(`[REQRESP_DEBUG] ${errorMsg}`);
      throw new Error(errorMsg);
    }

    const nodesToTurnOffTxGossip = Array.from({ length: NUM_VALIDATORS }, (_, i) => i).filter(
      i => !proposerIndexes.includes(i),
    );

    const fnEnd = Date.now();
    t.logger.info(`[REQRESP_DEBUG] getProposerIndexes completed in ${fnEnd - fnStart}ms`);
    t.logger.info(
      `[REQRESP_DEBUG] ProposerIndexes: ${proposerIndexes.join(', ')}, nodesToTurnOffTxGossip: ${nodesToTurnOffTxGossip.join(', ')}`,
    );

    return { proposerIndexes, nodesToTurnOffTxGossip };
  }
});
