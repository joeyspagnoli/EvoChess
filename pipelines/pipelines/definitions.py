from dagster import Definitions, load_assets_from_modules
from .resources import s3_resource
from . import assets as assets_module
# noqa: TID252

all_assets = load_assets_from_modules([assets_module])

defs = Definitions(
    assets=all_assets,
    resources={"s3": s3_resource},
)
