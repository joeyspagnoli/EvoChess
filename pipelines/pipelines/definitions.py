from dagster import Definitions, load_assets_from_modules
from .assets import chess_game_grab
from .resources import s3_resource
from pipelines import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=[chess_game_grab],
    resources={
        # This is the part you need to add.
        # You should configure it with the AWS region your S3 bucket is in.
        "s3": s3_resource
    },
)
