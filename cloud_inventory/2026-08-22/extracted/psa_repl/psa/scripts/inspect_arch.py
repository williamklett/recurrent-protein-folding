"""Inspect ESMFold architecture without downloading the 7 GB of weights.

Loads only the config + model skeleton (random init) and prints the module tree.
We use this to find the folding-trunk block class so we can hook activations later.
"""
from transformers import EsmConfig, EsmForProteinFolding


def main() -> None:
    cfg = EsmConfig.from_pretrained("facebook/esmfold_v1")
    print("=== Top-level config ===")
    for k in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
    ):
        print(f"  {k}: {getattr(cfg, k, None)}")

    print("\n=== ESMFold-specific config ===")
    esmfold_cfg = getattr(cfg, "esmfold_config", None)
    print(esmfold_cfg)

    # Random-init the model purely to inspect the module tree. No weights downloaded.
    print("\n=== Building random-init model skeleton (no weights) ===")
    model = EsmForProteinFolding(cfg)

    print("\n=== Folding trunk modules ===")
    trunk = model.esm.esmfold.trunk if hasattr(model.esm, "esmfold") else model.trunk
    print(trunk)

    print("\n=== Top-level named children ===")
    for name, _ in model.named_children():
        print(f"  {name}")


if __name__ == "__main__":
    main()
