"""设备异常隔离与解除测试。

覆盖：隔离需提交匹配链头（head_mismatch 409）、隔离期间上传/确认/归档/
租约领取/租约完成/批次判定均 409（device_quarantined）而查询、链核验、
消费者登记与过期租约回收仍可用、同键幂等重试与同键异载荷 409、重复隔离
409、解除的前置条件（未隔离 409、链头变化 409、范围核验失败 409、未完成
租约 409）、解除成功恢复原流程、解除幂等、状态查询与不可变事件日志、
空链隔离/解除，以及并发隔离/解除不越过状态。
"""
import os
import sqlite3
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

THRESHOLDS = {"tolerance": 0.5, "max_drift": 5e-4, "max_missed": 3, "max_false": 3}
DEVICE = "dev-01"


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        c.db_path = str(tmp_path / "test.db")  # 便于直接篡改库验证解除前置条件
        yield c


def obs(n, base=1_700_000_000.0, step=0.5):
    return [
        {"local_id": f"o{i:03d}", "captured_at": base + i * step}
        for i in range(n)
    ]


def batch(batch_id, sequence, previous_digest="", observations=None, **kw):
    body = {
        "batch_id": batch_id,
        "sequence": sequence,
        "previous_digest": previous_digest,
        "observations": observations if observations is not None else obs(3),
        **THRESHOLDS,
    }
    body.update(kw)
    return body


def upload_chain(client, device, n):
    """上传 n 个链式批次，返回摘要列表（下标 0 = seq1）。"""
    digests = []
    prev = ""
    for seq in range(1, n + 1):
        r = client.post(
            f"/field-devices/{device}/batches",
            json=batch(f"b-{seq}", seq, previous_digest=prev),
        )
        assert r.status_code == 201, r.text
        digests.append(r.json()["digest"])
        prev = digests[-1]
    return digests


def register(client, device, consumer_id):
    return client.post(
        f"/field-devices/{device}/consumers", json={"consumer_id": consumer_id}
    )


def quarantine(client, device, head_sequence, head_digest, reason="anomaly", key="q-1"):
    return client.post(
        f"/field-devices/{device}/quarantine",
        json={
            "head_sequence": head_sequence,
            "head_digest": head_digest,
            "reason": reason,
            "idempotency_key": key,
        },
    )


def release(client, device, frm, to, key="r-1"):
    return client.post(
        f"/field-devices/{device}/quarantine/release",
        json={"from_sequence": frm, "to_sequence": to, "idempotency_key": key},
    )


def quarantine_last(client, device, n, digests, key="q-1"):
    """以当前链头隔离设备，断言 201 并返回响应体。"""
    r = quarantine(client, device, n, digests[-1] if digests else "", key=key)
    assert r.status_code == 201, r.text
    return r.json()


def tamper(client, sql, params=()):
    """直接改库模拟离线事故；解除前置条件必须能识别。"""
    conn = sqlite3.connect(client.db_path)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


# ---------------- 隔离：链头核对与幂等 ----------------


def test_quarantine_requires_matching_chain_head(client):
    digests = upload_chain(client, DEVICE, 3)

    # 链头序号不符
    r = quarantine(client, DEVICE, 2, digests[1])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "quarantine_conflict"
    assert detail["reason"] == "head_mismatch"
    assert detail["submitted_head_sequence"] == 2
    assert detail["current_head_sequence"] == 3
    assert detail["current_head_digest"] == digests[2]

    # 摘要不符（序号对、摘要错）
    r = quarantine(client, DEVICE, 3, "00" * 32)
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "head_mismatch"

    # 链头未被错误隔离改变：设备仍未隔离
    r = client.get(f"/field-devices/{DEVICE}/quarantine")
    assert r.json()["status"] == "normal"
    assert r.json()["quarantined"] is False


def test_quarantine_request_validation(client):
    upload_chain(client, DEVICE, 1)
    # head_sequence>=1 必须带摘要；空链头必须空摘要；原因/键不可空白
    bad_payloads = [
        {"head_sequence": 1, "head_digest": "", "reason": "r", "idempotency_key": "k"},
        {"head_sequence": 0, "head_digest": "ab" * 32, "reason": "r", "idempotency_key": "k"},
        {"head_sequence": 1, "head_digest": "zz" * 32, "reason": "r", "idempotency_key": "k"},
        {"head_sequence": 1, "head_digest": "ab" * 32, "reason": "  ", "idempotency_key": "k"},
        {"head_sequence": 1, "head_digest": "ab" * 32, "reason": "r", "idempotency_key": " "},
        {"head_sequence": -1, "head_digest": "", "reason": "r", "idempotency_key": "k"},
    ]
    for payload in bad_payloads:
        r = client.post(f"/field-devices/{DEVICE}/quarantine", json=payload)
        assert r.status_code == 422, payload


def test_quarantine_idempotent_replay_and_key_conflict(client):
    digests = upload_chain(client, DEVICE, 2)
    body = quarantine_last(client, DEVICE, 2, digests)
    assert body["status"] == "quarantined"
    assert body["quarantined"] is True
    event = body["event"]
    assert event["action"] == "quarantined"
    assert event["head_sequence"] == 2
    assert event["head_digest"] == digests[1]
    assert event["reason"] == "anomaly"

    # 同键同载荷重试：HTTP 200，返回原事件，不新增事件
    r = quarantine(client, DEVICE, 2, digests[1])
    assert r.status_code == 200
    assert r.json()["status"] == "already_quarantined"
    assert r.json()["event"]["event_id"] == event["event_id"]
    assert r.json()["quarantined"] is True

    # 同键不同载荷（原因不同）→ 409 idempotency_key_in_use
    r = quarantine(client, DEVICE, 2, digests[1], reason="changed", key="q-1")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "idempotency_key_in_use"
    assert detail["occupied_by_event"]["event_id"] == event["event_id"]

    # 事件日志仍只有一条
    r = client.get(f"/field-devices/{DEVICE}/quarantine/events")
    assert r.json()["count"] == 1


def test_already_quarantined_conflict(client):
    digests = upload_chain(client, DEVICE, 2)
    quarantine_last(client, DEVICE, 2, digests, key="q-1")

    # 换一个幂等键再次隔离 → 409 already_quarantined，附活动隔离事件
    r = quarantine(client, DEVICE, 2, digests[1], key="q-2")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "already_quarantined"
    assert detail["active_event"]["idempotency_key"] == "q-1"
    assert detail["active_event"]["reason"] == "anomaly"
    assert "quarantined_at" in detail


# ---------------- 隔离期间：变更类 409，只读类可用 ----------------


def test_quarantine_blocks_processing_keeps_readonly(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "worker-a")
    # 先领取一个租约（完成路径与回收路径都要验证）
    r = client.post(
        f"/field-devices/{DEVICE}/batch-leases",
        json={
            "consumer_id": "worker-a",
            "max_batches": 2,
            "lease_duration_seconds": 3600,
            "idempotency_key": "lease-1",
        },
    )
    assert r.status_code == 201
    lease_id = r.json()["lease"]["lease_id"]

    quarantine_last(client, DEVICE, 3, digests)

    # —— 变更类全部 409 device_quarantined ——
    r = client.post(
        f"/field-devices/{DEVICE}/batches",
        json=batch("b-4", 4, previous_digest=digests[2]),
    )
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "quarantine_conflict"
    assert detail["reason"] == "device_quarantined"
    assert detail["operation"] == "upload_batch"
    assert detail["quarantine_reason"] == "anomaly"
    assert detail["quarantined_head_sequence"] == 3

    r = client.post(
        f"/field-devices/{DEVICE}/acknowledgments",
        json={"consumer_id": "worker-a", "sequence": 1, "batch_digest": digests[0]},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "device_quarantined"
    assert r.json()["detail"]["operation"] == "acknowledge_batch"

    r = client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 1})
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "device_quarantined"

    r = client.post(
        f"/field-devices/{DEVICE}/batch-leases",
        json={
            "consumer_id": "worker-a",
            "max_batches": 1,
            "lease_duration_seconds": 60,
            "idempotency_key": "lease-2",
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "device_quarantined"

    r = client.post(
        f"/field-devices/{DEVICE}/batch-leases/{lease_id}/complete",
        json={
            "consumer_id": "worker-a",
            "items": [
                {"sequence": 1, "batch_digest": digests[0]},
                {"sequence": 2, "batch_digest": digests[1]},
            ],
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "device_quarantined"

    r = client.post(
        f"/field-devices/{DEVICE}/batch-judgments",
        json={"idempotency_key": "bj-1", "from_sequence": 1, "to_sequence": 3},
    )
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "device_quarantined"

    # —— 只读与运维类仍可用 ——
    assert client.get(f"/field-devices/{DEVICE}/cursor").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/batches").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/batches/1").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/archive-status").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/batch-leases").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/batch-leases/{lease_id}").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/quarantine").status_code == 200
    assert client.get(f"/field-devices/{DEVICE}/quarantine/events").status_code == 200

    r = client.post(
        f"/field-devices/{DEVICE}/verify-chain",
        json={"from_sequence": 1, "to_sequence": 3},
    )
    assert r.status_code == 200 and r.json()["ok"] is True

    # 消费者登记不处理批次，仍可用
    r = register(client, DEVICE, "late-consumer")
    assert r.status_code == 201

    # 过期租约回收仍可用（强制过期后回收）
    tamper(client, "UPDATE batch_leases SET expires_at = 1 WHERE id = ?", (lease_id,))
    r = client.post(
        f"/field-devices/{DEVICE}/batch-leases/{lease_id}/recover",
        json={"consumer_id": "worker-a"},
    )
    assert r.status_code == 201
    assert r.json()["status"] == "reclaimed"

    # 被拦截的写操作没有留下任何痕迹：链头、确认、归档水位均未变
    cursor = client.get(f"/field-devices/{DEVICE}/cursor").json()
    assert cursor["last_sequence"] == 3
    assert cursor["batch_count"] == 3
    status = client.get(f"/field-devices/{DEVICE}/archive-status").json()
    assert status["archived_through"] == 0
    assert status["consumers"][0]["watermark"] == 0


# ---------------- 解除：前置条件 ----------------


def test_release_not_quarantined(client):
    upload_chain(client, DEVICE, 2)
    r = release(client, DEVICE, 1, 2)
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "not_quarantined"


def test_release_success_restores_flow(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "worker-a")
    quarantine_last(client, DEVICE, 3, digests)

    r = release(client, DEVICE, 1, 3)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "released"
    assert body["quarantined"] is False
    assert body["event"]["action"] == "released"
    assert body["event"]["from_sequence"] == 1
    assert body["event"]["to_sequence"] == 3
    assert body["event"]["head_sequence"] == 3
    assert body["event"]["head_digest"] == digests[2]
    # 解除响应附本次范围核验结果
    assert body["verification"]["ok"] is True
    assert body["verification"]["checked_batches"] == 3

    # 状态恢复 normal，流程恢复
    r = client.get(f"/field-devices/{DEVICE}/quarantine")
    assert r.json()["status"] == "normal"
    r = client.post(
        f"/field-devices/{DEVICE}/batches",
        json=batch("b-4", 4, previous_digest=digests[2]),
    )
    assert r.status_code == 201


def test_release_idempotent_replay_and_key_conflict(client):
    digests = upload_chain(client, DEVICE, 2)
    quarantine_last(client, DEVICE, 2, digests)
    body = release(client, DEVICE, 1, 2).json()
    event = body["event"]

    # 同键同范围重试：200，返回原解除事件，不重复核验（verification 为空）
    r = release(client, DEVICE, 1, 2)
    assert r.status_code == 200
    assert r.json()["status"] == "already_released"
    assert r.json()["event"]["event_id"] == event["event_id"]
    assert r.json()["verification"] is None
    assert r.json()["quarantined"] is False

    # 同键不同范围 → 409 idempotency_key_in_use
    r = release(client, DEVICE, 1, 1)
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "idempotency_key_in_use"

    # 已解除后再用新键解除 → 409 not_quarantined
    r = release(client, DEVICE, 1, 2, key="r-2")
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "not_quarantined"

    # 事件日志：隔离 + 解除各一条，不可变
    r = client.get(f"/field-devices/{DEVICE}/quarantine/events")
    assert [e["action"] for e in r.json()["events"]] == ["quarantined", "released"]


def test_release_rejected_when_head_changed(client):
    digests = upload_chain(client, DEVICE, 2)
    quarantine_last(client, DEVICE, 2, digests)

    # 模拟离线事故：隔离期间链头被改写
    tamper(
        client,
        "UPDATE device_chains SET last_sequence = 3, head_digest = ? WHERE device_id = ?",
        ("ff" * 32, DEVICE),
    )
    r = release(client, DEVICE, 1, 2)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "head_changed"
    assert detail["quarantined_head_sequence"] == 2
    assert detail["quarantined_head_digest"] == digests[1]
    assert detail["current_head_sequence"] == 3
    # 仍处隔离中
    assert client.get(f"/field-devices/{DEVICE}/quarantine").json()["quarantined"] is True


def test_release_rejected_when_verification_fails(client):
    digests = upload_chain(client, DEVICE, 3)
    quarantine_last(client, DEVICE, 3, digests)

    # 篡改一批的保存摘要：范围核验必然发现 digest_mismatch
    tamper(
        client,
        "UPDATE device_batches SET digest = ? WHERE device_id = ? AND sequence = 2",
        ("00" * 32, DEVICE),
    )
    r = release(client, DEVICE, 1, 3)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "verification_failed"
    assert detail["issue_count"] > 0
    assert any(i["issue_code"] == "digest_mismatch" for i in detail["issues"])
    assert client.get(f"/field-devices/{DEVICE}/quarantine").json()["quarantined"] is True


def test_release_rejected_when_range_missing(client):
    digests = upload_chain(client, DEVICE, 2)
    quarantine_last(client, DEVICE, 2, digests)
    # 范围右端越过链头：核验无法覆盖 → verification_failed
    r = release(client, DEVICE, 1, 5)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "verification_failed"
    assert detail["verification_error"]["error"] == "batches_not_found"


def test_release_rejected_with_outstanding_leases(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "worker-a")
    r = client.post(
        f"/field-devices/{DEVICE}/batch-leases",
        json={
            "consumer_id": "worker-a",
            "max_batches": 2,
            "lease_duration_seconds": 3600,
            "idempotency_key": "lease-1",
        },
    )
    lease_id = r.json()["lease"]["lease_id"]
    quarantine_last(client, DEVICE, 3, digests)

    # 在持租约 → 409 outstanding_leases
    r = release(client, DEVICE, 1, 3)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["reason"] == "outstanding_leases"
    assert detail["leases"][0]["lease_id"] == lease_id
    assert detail["leases"][0]["expired"] is False

    # 过期但未回收的租约同样挡住解除
    tamper(client, "UPDATE batch_leases SET expires_at = 1 WHERE id = ?", (lease_id,))
    r = release(client, DEVICE, 1, 3)
    assert r.status_code == 409
    assert r.json()["detail"]["leases"][0]["expired"] is True

    # 隔离期间回收过期租约后解除成功
    r = client.post(
        f"/field-devices/{DEVICE}/batch-leases/{lease_id}/recover",
        json={"consumer_id": "worker-a"},
    )
    assert r.status_code == 201
    r = release(client, DEVICE, 1, 3)
    assert r.status_code == 201
    assert r.json()["status"] == "released"


def test_empty_chain_quarantine_and_release(client):
    # 空链：链头 (0, "")，隔离后无可核验批次，解除跳过范围核验
    r = quarantine(client, DEVICE, 0, "", reason="proactive", key="q-empty")
    assert r.status_code == 201
    assert r.json()["event"]["head_sequence"] == 0
    assert r.json()["event"]["head_digest"] == ""

    # 隔离期间上传被拦截
    r = client.post(f"/field-devices/{DEVICE}/batches", json=batch("b-1", 1))
    assert r.status_code == 409

    r = release(client, DEVICE, 1, 9, key="r-empty")
    assert r.status_code == 201
    assert r.json()["verification"] is None
    # 解除后可正常上传
    r = client.post(f"/field-devices/{DEVICE}/batches", json=batch("b-1", 1))
    assert r.status_code == 201


# ---------------- 状态查询与事件日志 ----------------


def test_quarantine_status_and_events(client):
    digests = upload_chain(client, DEVICE, 2)

    r = client.get(f"/field-devices/{DEVICE}/quarantine")
    assert r.json() == {
        "device_id": DEVICE,
        "status": "normal",
        "quarantined": False,
        "active_quarantine": None,
        "last_event": None,
        "event_count": 0,
    }

    quarantine_last(client, DEVICE, 2, digests, key="q-1")
    r = client.get(f"/field-devices/{DEVICE}/quarantine")
    body = r.json()
    assert body["status"] == "quarantined"
    assert body["quarantined"] is True
    assert body["active_quarantine"]["idempotency_key"] == "q-1"
    assert body["active_quarantine"]["reason"] == "anomaly"
    assert body["last_event"]["action"] == "quarantined"
    assert body["event_count"] == 1

    release(client, DEVICE, 1, 2, key="r-1")
    quarantine_last(client, DEVICE, 2, digests, key="q-2")
    r = client.get(f"/field-devices/{DEVICE}/quarantine")
    body = r.json()
    assert body["status"] == "quarantined"
    assert body["active_quarantine"]["idempotency_key"] == "q-2"
    assert body["event_count"] == 3

    # 事件日志按 id 升序、可分页；事件不可变（含各时刻链头快照）
    r = client.get(f"/field-devices/{DEVICE}/quarantine/events")
    events = r.json()["events"]
    assert [e["action"] for e in events] == ["quarantined", "released", "quarantined"]
    assert [e["idempotency_key"] for e in events] == ["q-1", "r-1", "q-2"]
    assert events[0]["head_digest"] == digests[1]
    assert events[1]["from_sequence"] == 1 and events[1]["to_sequence"] == 2
    assert r.json()["next_cursor"] is None

    r = client.get(f"/field-devices/{DEVICE}/quarantine/events", params={"limit": 2})
    assert r.json()["count"] == 2
    cursor = r.json()["next_cursor"]
    assert cursor == events[1]["event_id"]
    r = client.get(
        f"/field-devices/{DEVICE}/quarantine/events",
        params={"after_id": cursor, "limit": 2},
    )
    assert [e["idempotency_key"] for e in r.json()["events"]] == ["q-2"]


# ---------------- 并发：隔离/解除不越过状态 ----------------


def test_concurrent_quarantine_single_winner(client):
    digests = upload_chain(client, DEVICE, 2)
    results = []

    def worker(i):
        results.append(
            quarantine(client, DEVICE, 2, digests[1], key=f"q-{i}").status_code
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 恰好一个 201，其余全部 409 already_quarantined：不会越过状态
    assert results.count(201) == 1
    assert results.count(409) == 7
    r = client.get(f"/field-devices/{DEVICE}/quarantine/events")
    assert r.json()["count"] == 1


def test_concurrent_same_key_quarantine_single_event(client):
    digests = upload_chain(client, DEVICE, 2)
    results = []

    def worker():
        results.append(quarantine(client, DEVICE, 2, digests[1], key="q-same").status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 同键并发：恰好一个 201，其余 200 幂等重放，事件只有一条
    assert results.count(201) == 1
    assert results.count(200) == 7
    r = client.get(f"/field-devices/{DEVICE}/quarantine/events")
    assert r.json()["count"] == 1


def test_concurrent_release_single_winner(client):
    digests = upload_chain(client, DEVICE, 2)
    quarantine_last(client, DEVICE, 2, digests)
    results = []

    def worker(i):
        results.append(release(client, DEVICE, 1, 2, key=f"r-{i}").status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 恰好一个 201，其余全部 409 not_quarantined：解除不会越过状态
    assert results.count(201) == 1
    assert results.count(409) == 7
    r = client.get(f"/field-devices/{DEVICE}/quarantine/events")
    assert [e["action"] for e in r.json()["events"]] == ["quarantined", "released"]
