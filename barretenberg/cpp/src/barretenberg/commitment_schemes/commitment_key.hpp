// === AUDIT STATUS ===
// internal:    { status: Planned, auditors: [Sergei], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once

/**
 * @brief Provides interfaces for different 'CommitmentKey' classes.
 */

#include "barretenberg/common/bb_bench.hpp"
#include "barretenberg/common/ref_span.hpp"
#include "barretenberg/ecc/scalar_multiplication/scalar_multiplication.hpp"
#include "barretenberg/polynomials/polynomial.hpp"
#include "barretenberg/srs/factories/crs_factory.hpp"
#include "barretenberg/srs/global_crs.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <memory>
#include <string_view>

#ifdef BB_GPU_NATIVE
#include "barretenberg/gpu/commitment_schemes/commitment_key_msm.hpp"
#endif

namespace bb {
/**
 * @brief CommitmentKey object over a pairing group 𝔾₁.
 *
 * @details Commitments are computed as C = [p(x)] = ∑ᵢ aᵢ⋅Gᵢ where Gᵢ is the i-th element of the SRS. For BN254,
 * the SRS is given as a list of 𝔾₁ points { [xʲ]₁ }ⱼ where 'x' is unknown. For Grumpkin, they are random points. The
 * SRS stored in the commitment key is after applying the pippenger_point_table thus being double the size of what is
 * loaded from path.
 */
template <class Curve> class CommitmentKey {

    using Fr = typename Curve::ScalarField;
    using Commitment = typename Curve::AffineElement;

  protected:
    std::shared_ptr<srs::factories::Crs<Curve>> srs;

    static bool msm_shape_log_enabled()
    {
        static const bool enabled = [] {
            const char* value = std::getenv("BB_MSM_SHAPE_LOG");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        return enabled;
    }

    static void log_msm_shape(const char* kind, size_t length, size_t start_index, size_t chunk_size = 1)
    {
        if (!msm_shape_log_enabled()) {
            return;
        }
        std::fprintf(stderr,
                     "BB_MSM_SHAPE kind=%s length=%zu start=%zu chunk_size=%zu\n",
                     kind,
                     length,
                     start_index,
                     chunk_size);
    }

  public:
    size_t srs_size;

    CommitmentKey() = default;

    /**
     * @brief Construct a new Kate Commitment Key object from existing SRS
     *
     * @param num_points Number of points needed for commitments
     */
    CommitmentKey(const size_t num_points)
        : srs(srs::get_crs_factory<Curve>()->get_crs(num_points))
        , srs_size(num_points)
    {
#ifdef BB_GPU_NATIVE
        gpu::init_commitment_key_srs<Curve>(get_monomial_points());
#endif
    }
    /**
     * @brief Checks the commitment key is properly initialized.
     *
     * @return bool
     */
    bool initialized() const { return srs != nullptr; }

    std::span<Commitment> get_monomial_points() const { return srs->get_monomial_points(); }
    size_t get_monomial_size() const { return srs->get_monomial_size(); }

    /**
     * @brief Uses the ProverSRS to create a commitment to p(X)
     *
     * @param polynomial a univariate polynomial p(X) = ∑ᵢ aᵢ⋅Xⁱ
     * @return Commitment computed as C = [p(x)] = ∑ᵢ aᵢ⋅Gᵢ
     */
    Commitment commit(PolynomialSpan<const Fr> polynomial) const
    {
        BB_BENCH_NAME("CommitmentKey::commit");
        std::span<const Commitment> point_table = get_monomial_points();
        size_t consumed_srs = polynomial.start_index + polynomial.size();
        if (consumed_srs > get_monomial_size()) {
            throw_or_abort(format("Attempting to commit to a polynomial that needs ",
                                  consumed_srs,
                                  " points with an SRS of size ",
                                  get_monomial_size()));
        }
        log_msm_shape("commit", polynomial.size(), polynomial.start_index);
#ifdef BB_GPU_NATIVE
        if constexpr (gpu::commitment_key_msm_available<Curve>) {
            return gpu::commitment_key_msm<Curve>(polynomial, point_table);
        }
#endif
        return scalar_multiplication::pippenger_unsafe<Curve>(polynomial, point_table);
    };
    /**
     * @brief Batch commitment to multiple polynomials
     * @details Uses batch_multi_scalar_mul for more efficient processing when committing to multiple polynomials.
     *          The input polynomials are not const because batch_mul modifies them and then restores them back.
     *
     * @param polynomials vector of polynomial spans to commit to
     * @return std::vector<Commitment> vector of commitments, one for each polynomial
     */
    std::vector<Commitment> batch_commit(RefSpan<Polynomial<Fr>> polynomials,
                                         size_t max_batch_size = std::numeric_limits<size_t>::max()) const
    {
        BB_BENCH_NAME("CommitmentKey::batch_commit");

        std::vector<Commitment> commitments;

        for (size_t i = 0; i < polynomials.size();) {
            size_t batch_size = std::min(max_batch_size, polynomials.size() - i);
            size_t batch_end = i + batch_size;

            std::vector<std::span<const Commitment>> points_spans;
            std::vector<std::span<Fr>> scalar_spans;
            points_spans.reserve(batch_size);
            scalar_spans.reserve(batch_size);

            for (auto& polynomial : polynomials.subspan(i, batch_end - i)) {
                std::span<const Commitment> point_table = get_monomial_points().subspan(polynomial.start_index());
                size_t consumed_srs = polynomial.start_index() + polynomial.size();
                if (consumed_srs > get_monomial_size()) {
                    throw_or_abort(format("Attempting to commit to a polynomial that needs ",
                                          consumed_srs,
                                          " points with an SRS of size ",
                                          get_monomial_size()));
                }
                log_msm_shape("batch_commit", polynomial.size(), polynomial.start_index(), batch_size);
                scalar_spans.emplace_back(polynomial.coeffs());
                points_spans.emplace_back(point_table);
            }

#ifdef BB_GPU_NATIVE
            if constexpr (gpu::commitment_key_msm_available<Curve>) {
                auto results = gpu::commitment_key_batch_msm<Curve>(points_spans, scalar_spans);
                commitments.insert(commitments.end(), results.begin(), results.end());
            } else
#endif
            {
                auto results =
                    scalar_multiplication::MSM<Curve>::batch_multi_scalar_mul(points_spans, scalar_spans, false);
                commitments.insert(commitments.end(), results.begin(), results.end());
            }
            i += batch_size;
        }
        return commitments;
    };

    // helper builder struct for constructing a batch to commit at once
    struct CommitBatch {
        CommitmentKey* key;
        RefVector<Polynomial<Fr>> wires;
        std::vector<std::string> labels;
        std::vector<Commitment> commit_and_send_to_verifier(auto transcript,
                                                            size_t max_batch_size = std::numeric_limits<size_t>::max())
        {
            std::vector<Commitment> commitments = key->batch_commit(wires, max_batch_size);
            for (size_t i = 0; i < commitments.size(); ++i) {
                transcript->send_to_verifier(labels[i], commitments[i]);
            }

            return commitments;
        }

        void add_to_batch(Polynomial<Fr>& poly, const std::string& label, bool mask)
        {
            if (mask) {
                poly.mask();
            }
            wires.push_back(poly);
            labels.push_back(label);
        }
    };

    CommitBatch start_batch() { return CommitBatch{ this, {}, {} }; }
};

} // namespace bb
