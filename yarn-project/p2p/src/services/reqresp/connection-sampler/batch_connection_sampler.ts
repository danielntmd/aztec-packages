import { createLogger } from '@aztec/foundation/log';

import type { PeerId } from '@libp2p/interface';

import type { ConnectionSampler } from './connection_sampler.js';

/**
 * Manages batches of peers for parallel request processing.
 * Tracks active peers and provides deterministic peer assignment for requests.
 *
 * Example with 3 peers and 10 requests:
 *
 * Peers:    [P1]      [P2]     [P3]
 *           ↓ ↓ ↓ ↓   ↓ ↓ ↓   ↓ ↓ ↓
 * Requests: 0,1,2,9 | 3,4,5 | 6,7,8
 *
 * Each peer handles a bucket of consecutive requests.
 * If a peer fails, it is replaced while maintaining the same bucket.
 */
export class BatchConnectionSampler {
  private readonly batch: PeerId[] = [];
  private readonly requestsPerPeer: number;

  constructor(
    private readonly connectionSampler: ConnectionSampler,
    batchSize: number,
    maxPeers: number,
    exclude?: PeerId[],
    private readonly logger = createLogger('p2p:reqresp:batch-connection-sampler'),
  ) {
    this.logger.debug(`[REQRESP_DEBUG] BatchConnectionSampler creating batch`, {
      batchSize,
      maxPeers,
      excludeCount: exclude?.length || 0,
    });

    if (maxPeers <= 0) {
      throw new Error('Max peers cannot be 0');
    }
    if (batchSize <= 0) {
      throw new Error('Batch size cannot be 0');
    }

    // Calculate how many requests each peer should handle, cannot be 0
    this.requestsPerPeer = Math.max(1, Math.floor(batchSize / maxPeers));

    // Sample initial peers
    const excluding = exclude && new Map(exclude.map(peerId => [peerId.toString(), true] as const));
    const samplingStart = Date.now();
    this.batch = this.connectionSampler.samplePeersBatch(maxPeers, excluding);
    const samplingEnd = Date.now();

    this.logger.debug(`[REQRESP_DEBUG] BatchConnectionSampler initialized`, {
      sampledPeerCount: this.batch.length,
      requestsPerPeer: this.requestsPerPeer,
      samplingTimeMs: samplingEnd - samplingStart,
      peers: this.batch.map(p => p.toString()),
    });
  }

  /**
   * Gets the peer responsible for handling a specific request index
   *
   * @param index - The request index
   * @returns The peer assigned to handle this request
   */
  getPeerForRequest(index: number): PeerId | undefined {
    if (this.batch.length === 0) {
      return undefined;
    }

    // Calculate which peer bucket this index belongs to
    const peerIndex = Math.floor(index / this.requestsPerPeer) % this.batch.length;
    return this.batch[peerIndex];
  }

  /**
   * Removes a peer and replaces it with a new one, maintaining the same position
   * in the batch array to keep request distribution consistent
   *
   * @param peerId - The peer to remove and replace
   */
  removePeerAndReplace(peerId: PeerId): void {
    const index = this.batch.findIndex(p => p === peerId);
    if (index === -1) {
      this.logger.debug(`[REQRESP_DEBUG] Peer not found in batch for removal: ${peerId.toString()}`);
      return;
    }

    this.logger.debug(`[REQRESP_DEBUG] Removing and replacing peer at index ${index}`, {
      peerId: peerId.toString(),
      batchSizeBefore: this.batch.length,
    });

    const excluding = new Map([[peerId.toString(), true]]);
    const replacementStart = Date.now();
    const newPeer = this.connectionSampler.getPeer(excluding); // Q: Shouldn't we accumulate all excluded peers? Otherwise the sampler could return us a previously excluded peer?
    const replacementEnd = Date.now();

    if (newPeer) {
      this.batch[index] = newPeer;
      this.logger.debug(`[REQRESP_DEBUG] Replaced peer successfully in ${replacementEnd - replacementStart}ms`, {
        oldPeer: peerId.toString(),
        newPeer: newPeer.toString(),
        index,
        batchSizeAfter: this.batch.length,
      });
    } else {
      // If we couldn't get a replacement, remove the peer and compact the array
      this.batch.splice(index, 1);
      this.logger.warn(
        `[REQRESP_DEBUG] No replacement peer available, removed peer from batch in ${replacementEnd - replacementStart}ms`,
        {
          removedPeer: peerId.toString(),
          index,
          batchSizeBefore: this.batch.length + 1,
          batchSizeAfter: this.batch.length,
        },
      );
    }
  }

  /**
   * Gets the number of active peers
   */
  get activePeerCount(): number {
    return this.batch.length;
  }

  /**
   * Gets the number of requests each peer is assigned to handle
   */
  get requestsPerBucket(): number {
    return this.requestsPerPeer;
  }
}
