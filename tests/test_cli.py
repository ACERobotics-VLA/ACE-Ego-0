from unittest.mock import Mock

import pytest

from evaluation.robocasa24 import cli


def test_wait_for_server_uses_websocket_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(cli.websockets.sync.client, "connect", Mock(return_value=connection))

    cli._wait_for_server("127.0.0.1", 5678, process, timeout=1)

    connection.recv.assert_called_once_with(timeout=1)
    cli.websockets.sync.client.connect.assert_called_once_with(
        "ws://127.0.0.1:5678",
        compression=None,
        proxy=None,
        open_timeout=1,
        close_timeout=1,
    )


def test_wait_for_server_reports_early_server_exit() -> None:
    process = Mock(returncode=17)
    process.poll.return_value = 17

    with pytest.raises(RuntimeError, match="code 17"):
        cli._wait_for_server("127.0.0.1", 5678, process, timeout=1)


def test_server_command_targets_the_local_host() -> None:
    contract = Mock(weights_path="/tmp/checkpoints/model.pt")

    command = cli._server_command(contract, "127.0.0.1", 5678)

    assert "--host" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "5678"
