"""CI-only persistence acceptance for both release StatefulSets in disposable Kind."""

import json
import os
import secrets
import subprocess

from scripts.release_host.backends import Kubernetes
from scripts.release_images import UPSTREAM

NAME = "pmmc-ci"
NS = "pmmc-storage-test"
CONTEXT = "kind-pmmc-release-ci"
ROLES = ("postgres", "minio")


def service_backend(images):
    return Kubernetes(
        {
            "instance": NAME,
            "images": images,
            "cluster": {
                "context": CONTEXT,
                "namespace": NS,
                "storage_class": "standard",
                "postgres_storage_gib": 1,
                "minio_storage_gib": 1,
            },
            "resources": {r: {"cpu": 1, "memory_gib": 1} for r in ROLES},
        },
        NAME,
        None,
        {},
        None,
    )


def kube(*args, obj=None):
    return subprocess.check_output(
        ["kubectl", "--context", CONTEXT, "-n", NS, *args],
        input=None if obj is None else json.dumps(obj),
        text=True,
        timeout=360,
    ).strip()


def identity(kind, name):
    obj = json.loads(kube("get", kind, name, "-o", "json"))
    if kind == "pvc":
        if obj["status"]["phase"] != "Bound" or obj["spec"]["accessModes"] != ["ReadWriteOnce"]:
            raise ValueError(f"{name}: not a bound RWO claim")
        return (obj["metadata"]["uid"], obj["spec"]["volumeName"])
    return obj["metadata"]["uid"]


def sql(query):
    return kube(
        "exec",
        f"{NAME}-postgres-0",
        "--",
        "psql",
        "-XAt",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "pmmc_owner",
        "-d",
        "pmmc",
        "-c",
        query,
    )


def wait_final_database():
    # The entrypoint temporarily runs a socket-only initialization server.
    kube(
        "exec",
        f"{NAME}-postgres-0",
        "--",
        "/bin/sh",
        "-ec",
        "i=0; until pg_isready -q -h 127.0.0.1 -U pmmc_owner -d pmmc; do "
        'i=$((i+1)); test "$i" -lt 90; sleep 1; done',
    )


def storage(cmd):
    return kube(
        "exec",
        f"{NAME}-minio-0",
        "--",
        "/bin/sh",
        "-ec",
        'export MC_HOST_probe="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@127.0.0.1:9000"; '
        + cmd,
    )


def check():
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("KUBECONFIG"):
        raise RuntimeError("use only with the release CI job's disposable Kind kubeconfig")
    backend = service_backend(json.loads(UPSTREAM.read_bytes()))
    kube("create", "namespace", NS)
    secret = backend.obj("Secret", f"{NAME}-credentials")
    secret.update(
        immutable=True,
        stringData={
            k: secrets.token_hex(16)
            for k in (
                "PMMC_POSTGRES_ADMIN_PASSWORD",
                "PMMC_MINIO_ROOT_USER",
                "PMMC_MINIO_ROOT_PASSWORD",
            )
        },
    )
    kube("create", "-f", "-", obj=secret)
    services = [f"{NAME}-{r}" for r in ROLES]
    for role in ROLES:
        for obj in backend.service_objects(role):
            kube("create", "-f", "-", obj=obj)
    for name in services:
        kube("rollout", "status", f"statefulset/{name}", "--timeout=300s")
    wait_final_database()

    marker = secrets.token_hex(24)
    if sql("SELECT current_setting('server_version_num')::int / 10000") != "16":
        raise ValueError("not PostgreSQL 16")
    sql(
        "CREATE EXTENSION vector; CREATE TABLE persistence_marker (value text, v vector(2)); "
        f"INSERT INTO persistence_marker VALUES ('{marker}', '[1,2]')"
    )
    storage(f"mc mb probe/storage-test; printf '%s' '{marker}' | mc pipe probe/storage-test/marker")
    claims = [identity("pvc", n) for n in services]
    if claims[0][1] == claims[1][1]:
        raise ValueError("expected two distinct PVs")
    old = [identity("pod", n + "-0") for n in services]
    for name in services:
        kube("scale", f"statefulset/{name}", "--replicas=0")
        kube("wait", "--for=delete", f"pod/{name}-0", "--timeout=120s")
    for name in services:
        kube("scale", f"statefulset/{name}", "--replicas=1")
        kube("rollout", "status", f"statefulset/{name}", "--timeout=300s")
    wait_final_database()
    if (
        any(identity("pod", n + "-0") == uid for n, uid in zip(services, old, strict=True))
        or claims != [identity("pvc", n) for n in services]
        or sql("SELECT value FROM persistence_marker WHERE v = '[1,2]'::vector") != marker
        or storage("mc cat probe/storage-test/marker") != marker
    ):
        raise ValueError("replacement pods did not recover both durable markers from original PVCs")
    print("Both new pod UIDs read their original marker on the same two bound PVs.")


if __name__ == "__main__":
    check()
