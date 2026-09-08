from evaluation.robocasa24.tasks import ROBOCASA24_TASKS, select_tasks, task_env_name


def test_task_catalog_has_24_entries() -> None:
    assert len(ROBOCASA24_TASKS) == 24


def test_task_slice_and_environment_name() -> None:
    selected = select_tasks(2, 2)
    assert selected == ((2, "PnPMilkToMicrowaveClose"),)
    assert task_env_name(selected[0][1]).endswith("_GR1ArmsAndWaistFourierHands_Env")
