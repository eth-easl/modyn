
from experiments.yearbook.compare_trigger_policies.pipeline_config import (
    gen_pipeline_config,
)
from modyn.config.schema.pipeline import (
    DataAmountTriggerConfig,
    ModynPipelineConfig,
    OffsetEvalStrategyConfig,
    OffsetEvalStrategyModel,
    TimeTriggerConfig,
)
from modynclient.client.client import run_multiple_pipelines
from modynclient.config.schema.client_config import ModynClientConfig, Supervisor


def run_experiment() -> None:
    
    pipeline_configs: ModynPipelineConfig = []
    
    # time based triggers: every: 1y, 5y, 15y, 25y
    for years in [1, 5, 15, 25]:
        pipeline_configs.append(gen_pipeline_config(
            name=f"TimeTrigger_{years}y",
            trigger=TimeTriggerConfig(every=years, unit="y"),
            # as OffsetEvalStrategy uses time offsets, this is compliant with the trigger config
            eval_strategy=OffsetEvalStrategyModel(
                name="OffsetEvalStrategy",
                config=OffsetEvalStrategyConfig(
                    offsets=[f"{years}d"]
                )
            )
        ))
        
    # sample count based triggers: every: 100, 500, 1000, 2000, 10_000
    for count in [100, 500, 1000, 2000, 10_000]:
        pipeline_configs.append(gen_pipeline_config(
            name=f"DataAmountTrigger_{count}",
            trigger=DataAmountTriggerConfig(num_samples=count),
            eval_strategy=OffsetEvalStrategyModel(
                name="OffsetEvalStrategy",
                config=OffsetEvalStrategyConfig(
                    offsets=[f"{count}"]
                )
            )
        ))
        
    host = input("Enter the supervisors host address: ") or "localhost"
    port = int(input("Enter the supervisors port: ")) or 50063
    run_multiple_pipelines(
        client_config=ModynClientConfig(supervisor=Supervisor(ip=host, port=port)),
        pipeline_configs=pipeline_configs,
        start_replay_at=0,
        stop_replay_at=None,
        maximum_triggers=None
    )
    

if __name__ == "__main__":
    run_experiment()
