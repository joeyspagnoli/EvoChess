import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dagster import asset, AssetExecutionContext
from dagster_aws.s3 import S3Resource
import io
import zstandard
import chess.pgn


S3_BUCKET_NAME = "joseph-spagnoli-evochess"


@asset()
def chess_game_grab(context: AssetExecutionContext, s3: S3Resource):
    """
    Iterates through all available months in LiChess, decompresses the zst file in chunks
    filters for elo only above 2000, compresses again, writes to memory for the month.
    Once month is over, upload memory to S3 resource.
    """
    month_to_start = datetime.strptime("2013-03", "%Y-%m")
    max_months_to_check = 3
    # Start with the first day of the current month
    context.log.info("Starting the download process...")

    # Loop forward thru the months
    for i in range(max_months_to_check):
        # Go back i months from the current month (i=0 is last month)
        target_month_date = month_to_start + relativedelta(months=i + 1)
        year = target_month_date.year
        month = target_month_date.month

        url_date = f"{year}-{month:02d}"

        file_url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{url_date}.pgn.zst"

        context.log.info(
            f"Checking for {target_month_date.strftime('%B %Y')} data at: {file_url}"
        )

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
                        context.log.info(f"Processed {game_counter} games.")

                    if (
                        white_elo_str
                        and white_elo_str.isdigit()
                        and black_elo_str
                        and black_elo_str.isdigit()
                    ):
                        if (
                            int(game.headers["WhiteElo"]) > 2000
                            and int(game.headers["BlackElo"]) > 2000
                        ):
                            pgn_str = str(game)
                            full_pgn_record = pgn_str + "\n\n"

                            compressor.write(full_pgn_record.encode("utf-8"))

                            valid_games += 1
                        else:
                            continue
                    else:
                        continue

                compressor.close()
                in_memory_buffer.seek(0)

                s3_key = f"filtered-pgn/lichess_db_standard_rated_{url_date}.pgn.zst"

                context.log.info(
                    f"Adding {valid_games} valid games to filtered S3 bucket"
                )
                context.log.info(f"Uploading {s3_key} to S3 bucket {S3_BUCKET_NAME}...")

                s3.get_client().upload_fileobj(
                    Fileobj=in_memory_buffer,
                    Bucket=S3_BUCKET_NAME,
                    Key=s3_key,
                )

                context.log.info(f"✅ {url_date} Upload complete.")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                context.log.info("-> Not found. Trying previous month.")
                continue  # Go to the next iteration of the loop
            else:
                # For other errors (like 500 server error), we should stop
                context.log.info(f"An unexpected HTTP error occurred: {e}")
                return None
