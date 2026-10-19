"""codec.entropy extracted from the supplied v5.5 codec."""

from __future__ import annotations

import torch

from holozip.models.hyperprior import IndependentWaveletHyperprior


@torch.no_grad()
def make_cpu_entropy_model(
    model: IndependentWaveletHyperprior,
) -> IndependentWaveletHyperprior:
    """Build a CPU copy for deterministic entropy coding/decoding.

    CompressAI entropy models contain runtime CDF buffers whose shapes are empty
    immediately after construction.  A direct ``load_state_dict`` from an already
    updated model therefore fails with size-mismatch errors for buffers such as
    ``_quantized_cdf`` and ``_offset``.

    We copy only learned/model parameters first (including the learned
    EntropyBottleneck quantiles), deliberately skip the generated entropy-table
    buffers, and then rebuild those buffers on CPU with ``update(force=True)``.
    This also guarantees that compression and later reconstruction derive the
    entropy tables using the same CPU implementation.
    """
    cpu_model = IndependentWaveletHyperprior(model.cfg).cpu()

    source_state = model.state_dict()

    # These are generated/runtime buffers, not learned weights.  Their shapes are
    # empty in a freshly constructed CompressAI model, so they must be rebuilt
    # rather than loaded.
    generated_entropy_suffixes = (
        "._offset",
        "._quantized_cdf",
        "._cdf_length",
        ".scale_table",
    )

    learned_state = {
        key: value.detach().cpu()
        for key, value in source_state.items()
        if not key.endswith(generated_entropy_suffixes)
    }

    incompatible = cpu_model.load_state_dict(learned_state, strict=False)

    # Only generated entropy buffers are expected to be missing.
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            "Unexpected keys while creating CPU entropy model: " + ", ".join(unexpected)
        )

    allowed_missing_suffixes = generated_entropy_suffixes
    bad_missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith(allowed_missing_suffixes)
    ]
    if bad_missing:
        raise RuntimeError(
            "Missing learned parameters while creating CPU entropy model: "
            + ", ".join(bad_missing)
        )

    cpu_model.eval()

    # Rebuild EntropyBottleneck CDFs and GaussianConditional scale/CDF tables
    # from the copied learned parameters, entirely on CPU.
    cpu_model.update(force=True)

    # Defensive checks: the runtime tables must now exist.
    if cpu_model.entropy_bottleneck._quantized_cdf.numel() == 0:
        raise RuntimeError("CPU EntropyBottleneck CDF table was not initialized")
    if cpu_model.gaussian_conditional._quantized_cdf.numel() == 0:
        raise RuntimeError("CPU GaussianConditional CDF table was not initialized")
    if cpu_model.gaussian_conditional.scale_table.numel() == 0:
        raise RuntimeError("CPU GaussianConditional scale table was not initialized")

    return cpu_model
