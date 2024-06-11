import os

from experiments.arxiv.compare_trigger_policies.pipeline_config import gen_pipeline_config
from experiments.utils.experiment_runner import run_multiple_pipelines
from modyn.config.schema.pipeline import (
    AfterTrainingEvalTriggerConfig,
    DataAmountTriggerConfig,
    EvalHandlerConfig,
    IntervalEvalStrategyConfig,
    ModynPipelineConfig,
    PeriodicEvalTriggerConfig,
    TimeTriggerConfig,
    UntilNextTriggerEvalStrategyConfig,
)
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def run_experiment() -> None:
    pipeline_configs: list[ModynPipelineConfig] = []

    eval_handlers = [
        EvalHandlerConfig(
            name="scheduled+-1y",
            datasets=["arxiv_test"],
            strategy=IntervalEvalStrategyConfig(interval="(-1d, +1d)"),  # timestamps faked (y -> d)
            trigger=PeriodicEvalTriggerConfig(
                every="1d",
                start_timestamp=0,
                end_timestamp=1296000 + 1,  # 2022 - 2006 > 1.1.1970 - 16.1.1970
                matrix=True,
            ),
        ),
        EvalHandlerConfig(
            name="scheduled+-2y",
            datasets=["arxiv_test"],
            strategy=IntervalEvalStrategyConfig(interval="(-2d, +2d)"),
            trigger=PeriodicEvalTriggerConfig(
                every="1d",
                start_timestamp=0,
                end_timestamp=1296000 + 1,  # 2022 - 2006 > 1.1.1970 - 16.1.1970
                matrix=True,
            ),
        ),
        EvalHandlerConfig(
            name="scheduled+-3y",
            datasets=["arxiv_test"],
            strategy=IntervalEvalStrategyConfig(interval="(-3d, +3d)"),
            trigger=PeriodicEvalTriggerConfig(
                every="1d",
                start_timestamp=0,
                end_timestamp=1296000 + 1,  # 2022 - 2006 > 1.1.1970 - 16.1.1970
                matrix=True,
            ),
        ),
        EvalHandlerConfig(
            name="full",
            datasets=["arxiv_test"],
            strategy=UntilNextTriggerEvalStrategyConfig(),
            trigger=AfterTrainingEvalTriggerConfig(),
        ),
    ]

    # time based triggers: every: 1y, 2y, 5y, 10y
    for years in [1, 2, 5, 10]:
        pipeline_configs.append(
            gen_pipeline_config(
                name=f"timetrigger_{years}y",
                trigger=TimeTriggerConfig(every=f"{years}d"),
                eval_handlers=eval_handlers,
            )
        )

    # sample count based triggers: every: 1_000_000, 500_000, 200_000, 100_000, 50_000
    skip_data_amount_triggers = True
    if not skip_data_amount_triggers:
        for count in [1_000_000, 500_000, 200_000, 100_000, 50_000]:
            pipeline_configs.append(
                gen_pipeline_config(
                    name=f"dataamounttrigger_{count}",
                    trigger=DataAmountTriggerConfig(num_samples=count),
                    eval_handlers=eval_handlers,
                )
            )

    host = os.getenv("MODYN_SUPERVISOR_HOST")
    port = int(os.getenv("MODYN_SUPERVISOR_PORT", "0"))

    if not host:
        host = input("Enter the supervisors host address: ") or "localhost"
    if not port:
        port = int(input("Enter the supervisors port: ") or "50063")

    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=pipeline_configs,
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None,
    )


if __name__ == "__main__":
    run_experiment()
