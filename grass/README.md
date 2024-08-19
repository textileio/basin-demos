# Grass + Basin demo

[![PyPI](https://img.shields.io/pypi/v/grass.svg)](https://pypi.org/project/grass/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/dtbuchholz/grass/blob/main/LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg)](https://github.com/RichardLitt/standard-readme)

> Basin + Grass Reddit dataset with ML labeling

## Background

This project demonstrates how to use Basin object storage alongside compute workflows, focusing on the [Grass Reddit
dataset](https://huggingface.co/datasets/OpenCo7/UpVoteWeb) that includes user comment and post information. It
implements a simple ML model to attach labels to the dataset, which enhances the original data and can be used to train
a more sophisticated model.

### Classification

Below is a sample of the Grass dataset (with some truncated row values), which in its entirely is _quite massive_. To
facilitate this demonstration, we'll create a dataset that's considerably smaller—around 100 MiB.
| id | parent_id | post_id | text | url | date | author | subreddit | score | token_count | kind | language | language_score | media_urls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kg22zyv | t1_kfsxjid | 18vqe1t | ... | https://www.reddit.com/r/... | 2024-01-03T00:37:50 | ... | SquaredCircle | 1 | 21 | comment | en | 0.9606 | [] |

You can find a full description of all of the fields below (as defined [here](https://huggingface.co/datasets/OpenCo7/UpVoteWeb#dataset-structure)):

- `id` (_string_): A unique identifier for the comment or post.
- `parent_id` (_string_): The identifier of the parent comment or post. The prefixes are defined as follows:
  - _t5_: subreddit
  - _t3_: post
  - _t1_: comment
- `post_id` (_string_): The post identifier.
- `text` (_string_): The content of the comment or post, with email addresses and IP addresses anonymized.
- `url` (_string_): The URL of the original thread on Reddit.
- `date` (_string_): The timestamp of the comment or post in UTC.
- `author` (_string_): The username of the author of the comment or post.
- `subreddit` (_string_): The subreddit where the comment or post was made.
- `score` (_int_): The score (upvotes minus downvotes) of the comment or post.
- `token_count` (_int_): The number of tokens in the text, as determined by the GPT-2 tokenizer.
- `kind`: A comment or post.
- `language` (_string_): The detected language of the text.
- `language_score` (_float_): The confidence score of the language detection.
- `media_urls` (_string[]_): An array of links to any multimedia included in the comment or post.

Below are the topic classifications attached to the original dataset:

- **technology**
- **politics**
- **entertainment**
- **health**
- **education**
- **finance**
- **sports**
- **other**

Additionally, sentiment analysis is added for each comment or post in the dataset.

- **positive**
- **negative**
- **neutral**

### Implementation

The source data is first streamed, chunked, and converted into ~10 MiB parquet files. These are each stored in Basin as
an object under the key `grass/raw/chunk_<id>.parquet`. The data is then read back from Basin, and a simple ML model is applied

- **Sentiment**: Use HuggingFace `transformers` and [`distilbert/distilbert-base-uncased`](https://huggingface.co/blog/sentiment-analysis-python) or [similar](https://huggingface.co/mwkby/distilbert-base-uncased-sentiment-reddit-crypto).
- **Topic**: Use HuggingFace `transforms` and [`bart-large`](https://huggingface.co/facebook/bart-large) or [similar](https://github.com/pysentimiento/pysentimiento).

The resulting output will look something like this:

| id      | sentiment | topic         | subreddit    |
| ------- | --------- | ------------- | ------------ |
| kg22zyv | positive  | technology    | r/technology |
| kfrxfca | negative  | politics      | r/politics   |
| kfrxfda | neutral   | entertainment | r/movies     |

## Usage

### Setup

This project uses [poetry](https://python-poetry.org/docs/#installation) for dependency management. Make sure
[pipx](https://pipx.pypa.io/stable/installation/) is installed (e.g., `brew install pipx` on Mac) (or just use `pip`)
and then install poetry with `pipx install poetry`.

The core logic depends on pytorch, which requires Python 3.11 to be installed on your machine. On MacOS, you can do this
with `pyenv`:

```
brew install pyenv
pyenv install 3.11
pyenv global 3.11
```

This should make the python 3.11 version available to you (stored in the `.pyenv` folder), which you can then set with
poetry as the shell environment (e.g,. for python `3.11.9`):

```sh
poetry env use python3.11
```

Then, you'll want to create a new virtual environment and install dependencies:

```sh
poetry shell
poetry install
```

You'll also need to set up a Basin S3 adapter. It acts as a interface to the network and facilitates writing to a bucket
with a private key. The full instructions are [here](https://github.com/textileio/basin-s3), but you can build it with:

```sh
git clone https://github.com/textileio/basin-s3
cd basin-s3
cargo build
```

You can then install the binary with the following to run on your machine:

```sh
cargo install --path lib/basin_s3 --features="binary"
```

Before running this python code, you **must start the S3 adapter**, where `--private-key` is your private key that will
write data to the object store:

```sh
basin_s3 \
--private-key <your_private_key> \
--access-key S3EXAMPLEAK \
--secret-key S3EXAMPLESK
```

### Running the pipeline

There are a few different ways to run the classification script. You can run it with the default settings. This will
use the `distilbert-base-uncased` model for sentiment analysis and the `facebook/bart-large` model for topic analysis:

The `initialize.py` file will run through all of the steps. It will create an object store (i.e., it requires the
`basin_s3` binary is running, as described above) and fetch the Grass dataset from HuggingFace. All of these are written
to the bucket under the `raw/` key prefix. Then, we take the raw dataset and process it, adding labels for a topic and
sentiment.

```sh
poetry run poe initialize
```

Once you run the script, it should log something like the following:

```sh
Bucket created successfully: t2rte6lkdaj7xw3umgtd7iax5f2tnlpo5excmuwva
Uploaded chunk_000.parquet to Basin at key: chunk_000.parquet
Uploaded chunk_001.parquet to Basin at key: chunk_001.parquet
Uploaded chunk_002.parquet to Basin at key: chunk_002.parquet
Processing object with key: raw/chunk_000.parquet
Processing object with key: raw/chunk_001.parquet
Processing object with key: raw/chunk_002.parquet
Processing complete.

Check the Basin bucket for the processed data:
 - Bucket: t2rte6lkdaj7xw3umgtd7iax5f2tnlpo5excmuwva
 - Objects (at prefix): processed/
 - Example object: processed/chunk_000.parquet
For example, with the CLI:
adm os get --address t2rte6lkdaj7xw3umgtd7iax5f2tnlpo5excmuwva processed/chunk_000.parquet
```

> Note: The full Grass dataset is available on HuggingFace, and the initialization script is designed to emulate a flow
> where data like this already exists on Basin and can be fetched for further processing. In total, this demo bucket
> stores ~100MiB as parquet files, but it's definitely possible for the full dataset to be stored on Basin!

## Development

This project uses [poetry](https://python-poetry.org/docs/#installation) for dependency management. Make sure [pipx](https://pipx.pypa.io/stable/installation/) is installed (e.g., `brew install pipx` on Mac) and then install poetry with `pipx install poetry`.

Once that's set up, you can install the project dependencies and run various tasks via [`poe`](https://poethepoet.natn.io/poetry_plugin.html) or `poetry run`. You can view all of the available tasks by running `poetry run poe --help` or reviewing the `[tool.poe.tasks]` of the `pyproject.toml` file.

```sh
# Start your shell and install dependencies
poetry shell
poetry install

# Make sure git is initialized and set up pre-commit and pre-push hooks
poetry run poe init
# Or, run this separately
git init
poetry run poe pre-commit

# Run linters
poetry run poe lint

# Run tests
poetry run poe test
poetry run poe coverage

# Build the package & publish to PyPI
poetry build
poetry poe publish
```

## Contributing

PRs accepted.

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [`poetry` project template](https://github.com/dtbuchholz/cookiecutter-poetry). Small note: If editing the `README`, please conform to the
[standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

Apache-2.0, © 2024 Dan Buchholz
