import os
import requests
from datetime import datetime, timezone
from dagster import asset, AssetExecutionContext, MonthlyPartitionsDefinition, Output
from dagster_aws.s3 import S3Resource
import io
import zstandard
import chess.pgn
import pandas as pd
from botocore.exceptions import ClientError
from pathlib import Path

from train_model import train as train_evochess

S3_BUCKET_NAME = "joseph-spagnoli-evochess"

monthly_partitions = MonthlyPartitionsDefinition(
    start_date=datetime(2013, 3, 1, tzinfo=timezone.utc)
)


def s3_object_exists(s3: S3Resource, bucket: str, key: str) -> bool:
    """Return True if an object exists at s3://bucket/key, False if not."""
    client = s3.get_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # Anything else is a real error
        raise


@asset(partitions_def=monthly_partitions)
def filtered_pgn_zst(context: AssetExecutionContext, s3: S3Resource) -> Output[str]:
    """
    For a single partition month (e.g., '2013-04'), download the corresponding
    Lichess .pgn.zst file, filter to games with WhiteElo/BlackElo > 2000, recompress,
    and upload the filtered file to S3.

    If the filtered .pgn.zst already exists in S3 for this month, reuse it.

    Returns:
        The S3 URI (or key) where the filtered .pgn.zst is stored.
    """

    # Partition key is like '2013-04'
    partition_key = context.asset_partition_key_for_output()
    dt = datetime.strptime(partition_key, "%Y-%m-%d")
    year = dt.year
    month = dt.month

    url_date = f"{year}-{month:02d}"
    file_url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{url_date}.pgn.zst"

    s3_key = f"filtered-pgn/lichess_db_standard_rated_{url_date}.pgn.zst"
    s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"

    if s3_object_exists(s3, S3_BUCKET_NAME, s3_key):
        context.log.info(
            f"Found existing filtered PGN at {s3_uri}. "
            "Skipping download/filter and reusing existing object."
        )
        return Output(
            value=s3_uri,
            metadata={
                "s3_bucket": S3_BUCKET_NAME,
                "s3_key": s3_key,
                "skipped": True,
                "reason": "existing S3 object",
                "source_url": file_url,
            },
        )

    context.log.info(f"Downloading Lichess data for {url_date} from {file_url}")

    try:
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()

            dctx = zstandard.ZstdDecompressor()
            decompressor = dctx.stream_reader(r.raw)

            pgn_stream = io.TextIOWrapper(decompressor, encoding="utf-8")
            game_counter = 0
            valid_games = 0

            in_memory_buffer = io.BytesIO()
            cctx = zstandard.ZstdCompressor()
            compressor = cctx.stream_writer(in_memory_buffer, closefd=False)

            context.log.info(
                f"Found {url_date} in database, beginning game parsing now..."
            )

            while True:
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    break
                game_counter += 1

                white_elo_str = game.headers.get("WhiteElo")
                black_elo_str = game.headers.get("BlackElo")

                if game_counter % 1000 == 0:
                    context.log.info(f"Processed {game_counter} games...")

                if (
                    white_elo_str
                    and white_elo_str.isdigit()
                    and black_elo_str
                    and black_elo_str.isdigit()
                ):
                    if int(white_elo_str) > 2000 and int(black_elo_str) > 2000:
                        pgn_str = str(game)
                        full_pgn_record = pgn_str + "\n\n"
                        compressor.write(full_pgn_record.encode("utf-8"))
                        valid_games += 1

            compressor.close()
            in_memory_buffer.seek(0)

            s3_key = f"filtered-pgn/lichess_db_standard_rated_{url_date}.pgn.zst"
            context.log.info(
                f"Adding {valid_games} valid games to filtered S3 bucket at {s3_key}"
            )
            context.log.info(f"Uploading to S3 bucket {S3_BUCKET_NAME}...")

            s3.get_client().upload_fileobj(
                Fileobj=in_memory_buffer,
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
            )

            context.log.info(f"✅ Upload complete for {url_date}.")

            # Return S3 URI (or you can just return the key if you prefer)
            s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"

            return Output(
                value=s3_uri,
                metadata={
                    "s3_bucket": S3_BUCKET_NAME,
                    "s3_key": s3_key,
                    "valid_games": valid_games,
                    "total_games_processed": game_counter,
                    "source_url": file_url,
                },
            )

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # For a partition that genuinely doesn't exist yet, we should fail the run
            msg = f"Lichess data for {url_date} not found (404)."
            context.log.warning(msg)
            raise RuntimeError(msg)
        else:
            msg = f"Unexpected HTTP error while fetching {file_url}: {e}"
            context.log.error(msg)
            raise


@asset(partitions_def=monthly_partitions)
def training_chunk_feather(
    context: AssetExecutionContext, s3: S3Resource, filtered_pgn_zst: str
) -> Output[str]:
    """
    For a given month (partition):
      - Download the filtered .pgn.zst from S3
      - Stream-decompress and parse the PGNs
      - Build a Dataframe with columns ["AN", "WhiteElo"]
      - Save it as a feather file in S3
      - Return the feather S3 URI

    If the feather file already exists in S3, reuse it.
    """

    partition_key = context.asset_partition_key_for_output()  # e.g. "2013-03-01"
    ym = partition_key[:7]  # "YYYY-MM"
    context.log.info(f"Building training dataframe for partition {partition_key}")

    if not filtered_pgn_zst.startswith("s3://"):
        raise ValueError(f"Unexpected filtered_pgn_zst URI: {filtered_pgn_zst}")

    _, _, bucket_and_key = filtered_pgn_zst.partition("s3://")
    bucket, _, _filtered_key = bucket_and_key.partition("/")

    context.log.info(f"Downloading filtered PGN from {filtered_pgn_zst}")

    feather_key = f"training-feather/evochess_{ym}.feather"
    feather_s3_uri = f"s3://{bucket}/{feather_key}"

    if s3_object_exists(s3, bucket, feather_key):
        context.log.info(
            f"Found existing training feather at {feather_s3_uri}. "
            "Skipping PGN parsing and reusing existing object."
        )
        return Output(
            value=feather_s3_uri,
            metadata={
                "partition_key": partition_key,
                "source_pgn_s3_uri": filtered_pgn_zst,
                "feather_s3_uri": feather_s3_uri,
                "skipped": True,
                "reason": "existing S3 object",
            },
        )

    _, _, bucket_and_key = filtered_pgn_zst.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")

    context.log.info(f"Downloading filtered PGN from {filtered_pgn_zst}")

    obj = s3.get_client().get_object(Bucket=bucket, Key=key)
    compressed_bytes = obj["Body"].read()
    compressed_stream = io.BytesIO(compressed_bytes)

    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(compressed_stream) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        rows = []
        game_counter = 0

        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            game_counter += 1

            white_elo = game.headers.get("WhiteElo")
            if not (white_elo and white_elo.isdigit()):
                continue

            # Build SAN move list for this game
            board = game.board()
            san_moves = []
            for move in game.mainline_moves():
                san_moves.append(board.san(move))
                board.push(move)

            # "AN" string is the full game in SAN, space-separated
            an_string = " ".join(san_moves)

            rows.append(
                {
                    "AN": an_string,
                    "WhiteElo": int(white_elo),
                }
            )

            if game_counter % 1000 == 0:
                context.log.info(
                    f"Parsed {game_counter} games from PGN; {len(rows)} rows so far..."
                )

    df = pd.DataFrame(rows)
    context.log.info(
        f"Finished parsing PGNs for {partition_key}: {len(df)} rows in DataFrame."
    )

    # Save DataFrame as feather back to S3
    ym = partition_key[:7]  # "YYYY-MM"
    feather_key = f"training-feather/evochess_{ym}.feather"

    buf = io.BytesIO()
    df.to_feather(buf)
    buf.seek(0)

    s3.get_client().put_object(
        Bucket=bucket,
        Key=feather_key,
        Body=buf.getvalue(),
    )

    feather_s3_uri = f"s3://{bucket}/{feather_key}"
    context.log.info(f"Uploaded training feather to {feather_s3_uri}")

    return Output(
        value=feather_s3_uri,
        metadata={
            "partition_key": partition_key,
            "source_pgn_s3_uri": filtered_pgn_zst,
            "feather_s3_uri": feather_s3_uri,
            "num_rows": len(df),
            "num_games_parsed": game_counter,
        },
    )


@asset(partitions_def=monthly_partitions)
def train_evochess_model(
    context: AssetExecutionContext,
    s3: S3Resource,
    training_chunk_feather: str,  # S3 URI to feather file
) -> Output[str]:
    """
    Training asset:

    - Downloads the monthly feather file from S3
    - Loads it as a DataFrame
    - Writes a local CSV (just for the training function)
    - Calls train(csv_path, model_out)
    - Returns the local model path and logs MLflow info in metadata
    """
    partition_key = context.asset_partition_key_for_output()  # e.g. "2013-03-01"
    ym = partition_key[:7]  # "YYYY-MM"

    if not training_chunk_feather.startswith("s3://"):
        raise ValueError(f"Unexpected training_chunk_csv URI: {training_chunk_feather}")

    # Parse S3 URI
    _, _, bucket_and_key = training_chunk_feather.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")

    context.log.info(f"Downloading training feather from {training_chunk_feather}")

    obj = s3.get_client().get_object(Bucket=bucket, Key=key)
    feather_bytes = obj["Body"].read()
    feather_buf = io.BytesIO(feather_bytes)

    # Load feather into DataFrame
    df = pd.read_feather(feather_buf)
    context.log.info(f"Loaded DataFrame with {len(df)} rows for {partition_key}")

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    local_csv_path = output_dir / f"evochess_{ym}.csv"
    df.to_csv(local_csv_path, index=False)
    context.log.info(f"Wrote local CSV to {local_csv_path}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    model_out = models_dir / f"evochess_{ym}_{timestamp}.pt"

    context.log.info(f"Starting training for partition {partition_key}")
    context.log.info(f"Model will be saved to {model_out}")

    result = train_evochess(csv_path=str(local_csv_path), model_out=str(model_out))

    run_id = result.get("run_id")
    final_test_accuracy = result.get("final_test_accuracy")
    # Normalize to string in case the training function returns a path or None
    model_path = result.get("model_path", str(model_out))
    model_path = str(model_path)

    context.log.info(f"MLflow run_id: {run_id}")
    context.log.info(f"Final test accuracy: {final_test_accuracy}")
    context.log.info(f"Model saved at: {model_path}")

    return Output(
        value=model_path,
        metadata={
            "partition_key": partition_key,
            "training_feather_s3_uri": training_chunk_feather,
            "local_csv_path": str(local_csv_path),
            "mlflow_run_id": run_id,
            "final_test_accuracy": final_test_accuracy,
            "model_path": model_path,
        },
    )
