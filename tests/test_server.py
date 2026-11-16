import asyncio
import base64
import json
import os

from aiohttp.test_utils import TestClient, TestServer
import pytest

from contract import TTL_S
from self_destruct import IdleRetirement
from server import JobManager, make_app
from conftest import audio_bytes, connector, encoded_image, payload
from test_engine import fake_backend


@pytest.fixture
async def app_client(tmp_path):
    now = [0.0]
    gate = tmp_path / "release"
    backend = fake_backend(env={**os.environ, "AVATAR_TEST_GATE": str(gate)})
    retirement = IdleRetirement(lambda _: None, env={}, clock=lambda: now[0])
    manager = JobManager(backend, tmp_path / "jobs", retirement, clock=lambda: now[0])
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        yield http, manager, backend, now, gate


async def start(http, **kwargs):
    response = await http.post("/clip/start", json=payload(**kwargs))
    assert response.status == 200, await response.text()
    body = await response.json()
    assert body["state"] == "queued"
    return body["id"]


async def wait_state(http, job_id, expected):
    for _ in range(300):
        response = await http.post("/clip/state", json={"id": job_id})
        data = await response.json()
        if data.get("state") == expected:
            return data
        await asyncio.sleep(.03)
    pytest.fail(f"job did not reach {expected}: {data}")


async def test_queue_one_running_two_waiting_and_real_file(app_client):
    http, manager, backend, _, gate = app_client
    first = await start(http)
    assert (await wait_state(http, first, "running"))["stage"] == "loading"
    second, third = await start(http), await start(http)
    assert manager.active == manager.retirement.jobs == 3
    response = await http.post("/clip/start", json=payload())
    assert response.status == 429 and await response.json() == {"error": "busy"}
    assert (await wait_state(http, second, "queued"))["progress"] == 0
    assert (await http.post("/clip/file", json={"id": third})).status == 409
    gate.touch()
    done = await wait_state(http, third, "done")
    assert done == {"state": "done", "progress": 1.0, "stage": "done", "fps": 25,
                    "width": 32, "height": 24, "frames": 50, "segments": 1, "gpus": 2, "exec_s": 0.0}
    assert all(manager.jobs[i].state == "done" for i in (first, second))
    assert manager.active == manager.retirement.jobs == 0
    directory = manager.jobs[first].directory
    assert [p.name for p in directory.iterdir()] == ["result.mp4"]
    for _ in range(2):
        response = await http.post("/clip/file", json={"id": first}, headers={"Range": "bytes=0-2"})
        assert response.status == 200 and response.headers["Content-Type"] == "video/mp4"
        body = await response.read()
        assert len(body) == int(response.headers["Content-Length"]) == manager.jobs[first].file.stat().st_size
        assert body[4:8] == b"ftyp"
    assert backend.healthy


async def test_concurrent_idempotence_and_conflicts(app_client):
    http, manager, _, _, gate = app_client
    body = payload()
    responses = await asyncio.gather(*(http.post("/clip/start", json=body) for _ in range(6)))
    values = [await response.json() for response in responses]
    assert all(response.status == 200 for response in responses)
    assert len({v["id"] for v in values}) == 1 and manager.active == 1
    for key, value in {"prompt": "Other", "seed": 0, "size": "720p", "image": encoded_image(size=(24, 32)),
                       "audio": base64.b64encode(audio_bytes(2.2)).decode()}.items():
        response = await http.post("/clip/start", json={**body, key: value})
        assert response.status == 409 and await response.json() == {"error": "conflict"}
    gate.touch()
    await wait_state(http, values[0]["id"], "done")
    response = await http.post("/clip/start", json=body)
    assert await response.json() == {"id": values[0]["id"], "state": "done"}
    assert not list(manager.root.glob(".validate-*"))


async def test_expiry_exact_boundary_request_reusable_and_fetch_lease(app_client):
    http, manager, _, now, gate = app_client
    gate.touch()
    request = "f" * 32
    job_id = await start(http, request=request)
    await wait_state(http, job_id, "done")
    directory = manager.jobs[job_id].directory
    now[0] = TTL_S - .001
    assert (await http.post("/clip/state", json={"id": job_id})).status == 200
    manager.jobs[job_id].fetching = True
    now[0] = TTL_S
    manager.expire()
    assert directory.exists()
    assert (await http.post("/clip/file", json={"id": job_id})).status == 404
    manager.jobs[job_id].fetching = False
    assert (await http.post("/clip/file", json={"id": job_id})).status == 404
    assert not directory.exists()
    assert await start(http, request=request) != job_id


async def test_failed_job_is_clean_and_next_job_usable(app_client):
    http, manager, backend, now, gate = app_client
    backend.env["AVATAR_TEST_MODE"] = "failure"
    first = await start(http)
    failed = await wait_state(http, first, "failed")
    assert failed["error"] == "generation failed" and "exec_s" not in failed
    assert list(manager.jobs[first].directory.iterdir()) == []
    assert (await http.post("/clip/file", json={"id": first})).status == 409
    assert backend.healthy
    backend.env["AVATAR_TEST_MODE"] = "normal"
    gate.touch()
    second = await start(http)
    await wait_state(http, second, "done")
    now[0] += TTL_S
    manager.expire()
    assert first not in manager.jobs and second not in manager.jobs


async def test_health_retirement_and_running_ttl(app_client):
    http, manager, backend, now, _ = app_client
    now[0] = 100
    assert (await http.get("/health")).status == 200
    assert manager.retirement.last_activity == 0
    first, second = await start(http), await start(http)
    now[0] = 10000
    manager.expire()
    assert first in manager.jobs and second in manager.jobs
    manager.retirement.retiring = True
    assert (await http.get("/health")).status == 503
    assert (await http.post("/clip/start", json=payload())).status == 429
    manager.retirement.retiring = False
    backend.healthy = False
    assert (await http.post("/clip/start", json=payload())).status == 429
    backend.healthy = True


@pytest.mark.parametrize("route", ["start", "state", "file"])
async def test_bad_json_unknown_keys_and_ids(app_client, route):
    http, *_ = app_client
    for data in (b"{", b"[]", b'{"id":"aaa","id":"bbb"}', b'{"id":null}', b'{"unknown":1}',
                 b'{"id":NaN}', b'{"id":"../../path"}'):
        response = await http.post("/clip/" + route, data=data)
        assert response.status == 400 and (await response.json())["error"] == "invalid"
    if route != "start":
        assert (await http.post("/clip/" + route, json={"id": "a" * 32})).status == 404


async def test_body_limit_and_machine_facts(tmp_path, monkeypatch):
    monkeypatch.setenv("AVATAR_MACHINE", json.dumps({"ram_gb": 128, "boot_s": 500, "secret": "fake"}))
    manager = JobManager(fake_backend(), tmp_path / "jobs", IdleRetirement(lambda _: None, env={}),
                         benchmark_id="a" * 32)
    async with TestClient(TestServer(make_app(manager)), connector=connector()) as http:
        response = await http.post("/clip/state", json={"id": "a" * 32})
        assert (await response.json())["machine"] == {"ram_gb": 128, "boot_s": 500}
        http.server.app._client_max_size = 128
        response = await http.post("/clip/start", data=b"x" * 129)
        assert response.status == 400 and (await response.json())["error"] == "invalid"
