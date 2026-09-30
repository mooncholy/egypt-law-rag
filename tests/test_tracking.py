import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from raglaw import tracking
from raglaw.schema import SCHEMA_VERSION, LogEvent
from raglaw.tracking import TrackingConfigError, git_state, sha256_file, stage_run

pytestmark = pytest.mark.unit


def artifact_log(client, run_id) -> list[dict]:
    """The JSONL log attached to a run, one dict per line."""
    [entry] = client.list_artifacts(run_id, "logs")
    local = Path(client.download_artifacts(run_id, entry.path))
    return [json.loads(line) for line in local.read_text("utf-8").splitlines()]


def events(lines: list[dict]) -> list[str]:
    return [line["event_type"] for line in lines if "event_type" in line]


def test_run_records_inputs_metrics_and_log(tracking_settings, mlflow_client):
    with stage_run("profile", input_hash="abc123", settings=tracking_settings) as run:
        run.log_metrics({"pages_total": 170})

    recorded = mlflow_client.get_run(run.run_id)
    assert recorded.info.status == "FINISHED"
    assert recorded.info.run_name == "profile"
    assert recorded.data.params == {
        "stage": "profile",
        "git_sha": git_state()[0],
        "schema_version": SCHEMA_VERSION,
        "input_hash": "abc123",
    }
    assert recorded.data.tags["stage"] == "profile"
    assert recorded.data.tags["git_dirty"] in {"true", "false"}
    assert recorded.data.metrics == {"pages_total": 170.0}
    assert events(artifact_log(mlflow_client, run.run_id)) == [
        LogEvent.STAGE_START,
        LogEvent.STAGE_COMPLETED,
    ]


def test_failure_marks_run_failed_and_still_attaches_log(
    tracking_settings, mlflow_client
):
    with (
        pytest.raises(RuntimeError, match="boom"),
        stage_run("extract", input_hash="x", settings=tracking_settings) as run,
    ):
        raise RuntimeError("boom")

    assert mlflow_client.get_run(run.run_id).info.status == "FAILED"
    assert events(artifact_log(mlflow_client, run.run_id)) == [
        LogEvent.STAGE_START,
        LogEvent.STAGE_FAILED,
    ]


def test_interrupt_marks_run_killed(tracking_settings, mlflow_client):
    with (
        pytest.raises(KeyboardInterrupt),
        stage_run("extract", input_hash="x", settings=tracking_settings) as run,
    ):
        raise KeyboardInterrupt

    assert mlflow_client.get_run(run.run_id).info.status == "KILLED"


def test_artifacts_land_under_the_configured_root(tracking_settings, mlflow_client):
    with stage_run("profile", input_hash="x", settings=tracking_settings) as run:
        pass

    experiment = mlflow_client.get_experiment_by_name(
        tracking_settings.tracking.experiment
    )
    root = f"{tracking_settings.mlflow_artifact_root}/{tracking_settings.tracking.experiment}"
    assert experiment.artifact_location == root
    assert mlflow_client.get_run(run.run_id).info.artifact_uri.startswith(root)


def test_stages_of_one_repro_share_the_experiment(tracking_settings, mlflow_client):
    with stage_run("profile", input_hash="x", settings=tracking_settings) as first:
        pass
    with stage_run("extract", input_hash="y", settings=tracking_settings) as second:
        pass

    runs = [mlflow_client.get_run(r.run_id).info for r in (first, second)]
    assert runs[0].experiment_id == runs[1].experiment_id
    assert [r.status for r in runs] == ["FINISHED", "FINISHED"]


def test_experiment_with_another_artifact_location_fails_loudly(tracking_settings):
    with stage_run("profile", input_hash="x", settings=tracking_settings):
        pass
    moved = tracking_settings.model_copy(
        update={"mlflow_artifact_root": "s3://elsewhere"}
    )

    with (
        pytest.raises(TrackingConfigError, match="s3://elsewhere"),
        stage_run("profile", input_hash="x", settings=moved),
    ):
        pass


def test_deleted_experiment_fails_loudly(tracking_settings, mlflow_client):
    with stage_run("profile", input_hash="x", settings=tracking_settings):
        pass
    experiment = mlflow_client.get_experiment_by_name(
        tracking_settings.tracking.experiment
    )
    mlflow_client.delete_experiment(experiment.experiment_id)

    with (
        pytest.raises(TrackingConfigError, match="deleted"),
        stage_run("profile", input_hash="x", settings=tracking_settings),
    ):
        pass


def test_unset_artifact_root_keeps_mlflows_default(tracking_settings, mlflow_client):
    local = tracking_settings.model_copy(update={"mlflow_artifact_root": None})

    with stage_run("profile", input_hash="x", settings=local) as run:
        pass

    assert mlflow_client.get_run(run.run_id).info.status == "FINISHED"


def test_without_git_the_run_says_unknown(
    tracking_settings, mlflow_client, monkeypatch
):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(tracking.subprocess, "run", no_git)

    with stage_run("profile", input_hash="x", settings=tracking_settings) as run:
        pass

    recorded = mlflow_client.get_run(run.run_id)
    assert recorded.data.params["git_sha"] == "unknown"
    assert recorded.data.tags["git_dirty"] == "unknown"


def test_git_state_matches_head():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert git_state()[0] == head


@pytest.mark.parametrize("preset", [None, "someone-else"])
def test_aws_profile_is_exported_unless_already_set(
    tracking_settings, monkeypatch, preset
):
    if preset is None:
        monkeypatch.delenv("AWS_PROFILE", raising=False)
    else:
        monkeypatch.setenv("AWS_PROFILE", preset)
    with_profile = tracking_settings.model_copy(update={"aws_profile": "from-settings"})

    with stage_run("profile", input_hash="x", settings=with_profile):
        pass

    assert os.environ["AWS_PROFILE"] == (preset or "from-settings")


def test_sha256_file_matches_hashlib(tmp_path):
    data = b"civil code" * 200_000  # spans several 1 MiB chunks
    path = tmp_path / "input.bin"
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_clean_checkout_is_not_dirty(git_repo):
    assert git_state()[1] is False


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda repo: (repo / "src/stage.py").write_text("ROWS = 2\n"), id="modified"
        ),
        pytest.param(
            lambda repo: (repo / "src/new_stage.py").write_text("X = 1\n"),
            id="untracked",
        ),
        pytest.param(lambda repo: (repo / "src/stage.py").unlink(), id="deleted"),
    ],
)
def test_uncommitted_code_is_dirty(git_repo, change):
    change(git_repo)

    assert git_state()[1] is True


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda repo: (repo / "dvc.lock").write_text("md5: 2\n"), id="dvc-lock"
        ),
        pytest.param(
            lambda repo: (
                (repo / "src/__pycache__").mkdir(),
                (repo / "src/__pycache__/stage.pyc").write_bytes(b"\0"),
            ),
            id="gitignored",
        ),
    ],
)
def test_changes_outside_code_are_not_dirty(git_repo, change):
    change(git_repo)

    assert git_state()[1] is False
