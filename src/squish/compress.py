import torch
import typer
from pathlib import Path
from transformers import AutoModel
from .model_compressor import SVDCompressor

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m", help="Path to the model to compress."
    ),
    output_path: Path = typer.Option(
        ..., "--output-path", "-o", help="Path to save the compressed model."
    ),
    retention: float = typer.Option(
        0.55,
        "--retention",
        "-r",
        help="Target information retention (e.g., 0.55 for 55%).",
    ),
    shard_size: str = typer.Option(
        "2GB", "--shard-size", help="Maximum size of each saved model shard."
    ),
):
    """Compresses a 🤗 Transformers model using Quality-based SVD."""
    print("🎯 QUALITY-BASED SVD COMPRESSION")
    print("=" * 60)
    typer.echo(f"🎚️  Target information retention: {retention:.0%}")
    typer.echo(f"📂 Loading model from: {model_path}")

    # Load a generic model
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Instantiate and run the compressor
    compressor = SVDCompressor(target_retention=retention)
    compressed_model = compressor.compress_model(model)

    print("\n💾 Saving compressed model...")
    # Save the full model with sharding
    compressed_model.save_pretrained(
        output_path, max_shard_size=shard_size, safe_serialization=True
    )
    typer.secho(
        f"✅ Successfully saved compressed model to: {output_path}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
