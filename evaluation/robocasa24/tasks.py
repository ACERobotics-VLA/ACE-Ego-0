"""Canonical task catalog for the GR1 RoboCasa 24 benchmark."""

ROBOCASA24_TASKS: tuple[str, ...] = (
    "PnPCupToDrawerClose",
    "PnPPotatoToMicrowaveClose",
    "PnPMilkToMicrowaveClose",
    "PnPBottleToCabinetClose",
    "PnPWineToCabinetClose",
    "PnPCanToDrawerClose",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)


def task_env_name(task_name: str) -> str:
    """Return the registered RoboCasa environment name for one task."""
    if task_name not in ROBOCASA24_TASKS:
        raise ValueError(f"Unknown GR1 RoboCasa 24 task: {task_name!r}")
    return f"gr1_unified/{task_name}_GR1ArmsAndWaistFourierHands_Env"


def select_tasks(task_start: int, task_end: int) -> tuple[tuple[int, str], ...]:
    """Return an inclusive validated task-index slice."""
    task_count = len(ROBOCASA24_TASKS)
    if not 0 <= task_start < task_count:
        raise ValueError(f"task_start must be in [0, {task_count - 1}], got {task_start}.")
    if not task_start <= task_end < task_count:
        raise ValueError(f"task_end must be in [{task_start}, {task_count - 1}], got {task_end}.")
    return tuple(enumerate(ROBOCASA24_TASKS[task_start : task_end + 1], start=task_start))
