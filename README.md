# Squish 🦑

A command-line tool for compressing 🤗 Transformers models using Quality-based Singular Value Decomposition (SVD). This tool replaces `nn.Linear` layers with low-rank approximations to reduce model size while retaining a target percentage of the original model's "energy".

## Installation

To get started, clone the repository and install the project in editable mode. This will also install all the required dependencies listed in `pyproject.toml`.

```bash
# Clone the repository
git clone git@github.com:salsasteve/squish.git
cd squish

# Install the project in editable mode
pip install -e .
```

If you plan to run the test suite, make sure to install the testing dependencies as well:

```bash
pip install -e ".[test]"
```

## Usage

This project is managed as an installable package, and the main command-line interface is exposed through the `squish-compress` script (as defined in `pyproject.toml`).

### Basic Compression

To compress a model, you need to provide the path to the source model and a path for the output. The tool will use a default information retention of 55%.

```bash
squish-compress --model-path ./path/to/your-model --output-path ./path/to/compressed-model
```

### Advanced Compression

You can customize the compression by specifying the target information retention and the maximum shard size for the saved model.

```bash
squish-compress \
    --model-path ./path/to/your-model \
    --output-path ./path/to/compressed-model \
    --retention 0.75 \
    --shard-size 1GB
```

  - `--model-path` / `-m`: (Required) Path to the model you want to compress.
  - `--output-path` / `-o`: (Required) Path where the compressed model will be saved.
  - `--retention` / `-r`: (Optional) The target information retention. A value of `0.75` means the compressed model will retain 75% of the original's energy. Defaults to `0.55`.
  - `--shard-size`: (Optional) The maximum size for each model shard when saving. Defaults to `2GB`.

### Getting Help

For a full list of commands and options, you can use the `--help` flag.

```bash
squish-compress --help
```

## Development

### Running Tests

To run the test suite and generate a coverage report, first ensure you have installed the test dependencies, then run `pytest` from the root of the project directory.

```bash
# Install test dependencies if you haven't already
pip install -e ".[test]"

# Run the test suite
pytest
```

The test command is configured in `pyproject.toml` to automatically run with coverage reporting. A summary will be printed to the terminal, and a detailed HTML report will be generated in the `htmlcov/` directory.
