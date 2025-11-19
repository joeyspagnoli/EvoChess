from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    build_schedule_from_partitioned_job,
    AssetSelection,
    ScheduleDefinition,
)
from .resources import s3_resource
from . import assets as assets_module
from .assets import monthly_partitions
# noqa: TID252

all_assets = load_assets_from_modules([assets_module])

monthly_selection = AssetSelection.keys(
    "filtered_pgn_zst",
    "training_chunk_feather",
    "train_evochess_model",
)

evochess_monthly_job = define_asset_job(
    "evochess_monthly_job",
    selection=monthly_selection,
    partitions_def=monthly_partitions,
)

# Runs at 3:00 AM America/New_York for the "latest" month partition
evochess_monthly_schedule = build_schedule_from_partitioned_job(
    evochess_monthly_job, minute_of_hour=0, hour_of_day=3
)

baseline_job = define_asset_job(
    "baseline_evochess_job",
    selection=AssetSelection.keys("baseline_evochess_model"),
)

# Example: run at 3:30 AM on the 1st of each month
baseline_monthly_schedule = ScheduleDefinition(
    name="baseline_evochess_monthly_schedule",
    job=baseline_job,
    cron_schedule="30 3 1 * *",
    execution_timezone="America/New_York",
)

defs = Definitions(
    assets=all_assets,
    resources={"s3": s3_resource},
    schedules=[evochess_monthly_schedule, baseline_monthly_schedule],
)
