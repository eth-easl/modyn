from typing import Any, Optional

from modyn.supervisor.internal.grpc.enums import CounterAction, IdType, MsgType, PipelineStage


def pipeline_res_msg(pipeline_id: int = -1, exception: Optional[str] = None) -> dict[str, Any]:
    ret: dict[str, Any] = {"pipeline_id": pipeline_id}
    if exception is not None:
        ret["exception"] = exception
    return ret


def pipeline_stage_msg(
    stage: PipelineStage, msg_type: MsgType, submsg: Optional[dict[str, Any]] = None, log: bool = False
) -> dict[str, Any]:
    ret = {"stage": str(stage), "msg_type": str(msg_type), "log": log}
    if submsg is not None:
        ret[msg_type] = submsg

    return ret


def dataset_submsg(dataset_id: str) -> dict[str, str]:
    return {"id": dataset_id}


def id_submsg(id_type: IdType, id_num: int) -> dict[str, Any]:
    return {"id_type": str(id_type), "id": id_num}


def counter_submsg(action: CounterAction, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    ret: dict[str, Any] = {"action": str(action)}
    if params is not None:
        ret[f"{action}_params"] = params
    return ret


def exit_submsg(exitcode: int, exception: Optional[str] = None) -> dict[str, Any]:
    ret: dict[str, Any] = {"exitcode": exitcode}
    if exception is not None:
        ret["exception"] = exception
    return ret
