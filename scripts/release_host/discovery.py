"""Read-only single-node admission estimates, bounded again at Job submission."""

import copy
import json
import re
from decimal import Decimal

from .config import DNS, Error, check
from .quota import DEFAULTS, GPU_ROLES, quantity
from .topology import BOUNDS, normalize

FIELDS = {"cpu": "cpus", "memory": "memory_gib", "nvidia.com/gpu": "gpus"}


# Do not return unrelated Pods' arguments, env, annotation payloads or volume
# sources to the host process. Kube names cannot contain these delimiters.
def pod_projection():
    tab = r'{"\t"}'
    fields = [
        "{.metadata.namespace}",
        "{.metadata.labels.pmmc-release-instance}",
        "{.metadata.ownerReferences}",
        "{.status.phase}",
        "{.spec.nodeName}",
        "{.spec.overhead}",
        "{.spec.resources}",
    ]
    for field in ("containers", "initContainers"):
        fields.append("{range .spec." + field + '[*]}{.name}{"="}{.resources}{"|"}{end}')
    for field in ("containerStatuses", "initContainerStatuses"):
        fields.append(
            "{range .status."
            + field
            + '[*]}{.name}{"="}{.allocatedResources}{"="}{.resources.requests}{"|"}{end}'
        )
    return "jsonpath={range .items[*]}" + tab.join(fields) + r'{"#\n"}{end}'


def projected_pods(raw):
    pods = []

    def entries(field):
        return [x.split("=") for x in field.split("|") if x]

    for row in raw.splitlines():
        if not row.endswith("#"):
            raise ValueError("incomplete pod allocation record")
        ns, owner, refs, phase, node, overhead, resources, main, init, running, starting = row[
            :-1
        ].split("\t")
        spec = {
            "nodeName": node,
            "overhead": json.loads(overhead or "{}"),
            "resources": json.loads(resources or "{}"),
        }
        for key, declared, status in (
            ("containers", main, running),
            ("initContainers", init, starting),
        ):
            actual = {
                name: [json.loads(allocated or "{}"), json.loads(applied or "{}")]
                for name, allocated, applied in entries(status)
            }
            spec[key] = [
                {
                    "name": name,
                    "resources": json.loads(res or "{}"),
                    "observed": actual.get(name, []),
                }
                for name, res in entries(declared)
            ]
        pods.append(
            {
                "metadata": {
                    "namespace": ns,
                    "labels": {"pmmc-release-instance": owner},
                    "ownerReferences": json.loads(refs or "[]"),
                },
                "status": {"phase": phase},
                "spec": spec,
            }
        )
    return pods


def requests(spec, key):
    def container(c):
        res = c.get("resources", {})
        declared = quantity(res.get("requests", {}).get(key, res.get("limits", {}).get(key, 0)))
        return max(
            [declared, *(quantity(current.get(key, 0)) for current in c.get("observed", []))]
        )

    # Conservative for init containers, including restartable sidecars. Counting
    # all concurrent may underestimate free capacity but never promises it twice.
    used = sum(container(c) for c in (*spec.get("containers", []), *spec.get("initContainers", [])))
    pod_request = quantity(spec.get("resources", {}).get("requests", {}).get(key, 0))
    return max(used, pod_request) + quantity(spec.get("overhead", {}).get(key, 0))


def active(pod):
    return pod.get("status", {}).get("phase") not in ("Failed", "Succeeded")


def tolerated(taint, tolerations):
    if taint["effect"] == "PreferNoSchedule":
        return True
    return any(
        t["key"] == taint["key"]
        and t.get("effect", "") in ("", taint["effect"])
        and (t["operator"] == "Exists" or t.get("value", "") == taint.get("value", ""))
        for t in tolerations
    )


def held_service(cfg, pod):
    meta = pod["metadata"]
    if (
        meta.get("namespace") != cfg["cluster"]["namespace"]
        or meta.get("labels", {}).get("pmmc-release-instance") != cfg["instance"]
    ):
        return False
    return any(
        o.get("kind") == "StatefulSet"
        and o.get("controller") is True
        and o.get("name") in {cfg["instance"] + "-" + s for s in ("postgres", "minio")}
        for o in meta.get("ownerReferences", [])
    )


def remaining_quota(cfg, quotas, pods, *, services):
    result = {}
    for quota in quotas:
        spec, status = quota["spec"], quota.get("status", {})
        for key, limit in spec["hard"].items():
            resource = key.removeprefix("requests.").removeprefix("limits.")
            if resource not in FIELDS:
                continue
            check(
                key in status.get("used", {}) and key in status.get("hard", {}),
                "namespace ResourceQuota usage or effective hard limit unavailable",
            )
            hard = min(quantity(limit), quantity(status["hard"][key]))
            free = hard - quantity(status["used"][key])
            if services and not (spec.get("scopes") or spec.get("scopeSelector")):
                for pod in pods:
                    if held_service(cfg, pod):
                        # Our existing service has requests == limits. Only credit
                        # its main container, never unrelated sidecars.
                        for c in pod["spec"]["containers"]:
                            if c["name"] in ("postgres", "minio"):
                                field = "limits" if key.startswith("limits.") else "requests"
                                free += quantity(
                                    c.get("resources", {}).get(field, {}).get(resource, 0)
                                )
            result[resource] = min(result.get(resource, hard), free, hard)
    return result


def from_snapshot(cfg, nodes, pods, quotas, roles):
    roles = set(DEFAULTS if roles is None else roles)
    check(roles <= DEFAULTS.keys(), "unknown deployment phase in resource discovery")
    services = roles != {"analysis-report"}
    if services:
        roles.update(("postgres", "minio"))
    pods = [pod for pod in pods if active(pod)]
    headroom = remaining_quota(cfg, quotas, pods, services=services)
    policy = cfg["resource_policy"]
    eligible = []
    reasons = set()
    for node in nodes:
        meta, spec, status = node["metadata"], node.get("spec", {}), node.get("status", {})
        host = meta.get("labels", {}).get("kubernetes.io/hostname")
        if (
            not isinstance(host, str)
            or not re.fullmatch(rf"{DNS}(?:\.{DNS})*", host)
            or host != cfg["cluster"].get("node_hostname", host)
            or meta.get("labels", {}).get("kubernetes.io/os") != "linux"
            or any(
                meta.get("labels", {}).get(k) != v
                for k, v in cfg["cluster"]["node_selector"].items()
            )
            or meta.get("deletionTimestamp")
            or spec.get("unschedulable")
            or any(not tolerated(t, cfg["cluster"]["tolerations"]) for t in spec.get("taints", []))
            or not any(
                c["type"] == "Ready" and c["status"] == "True" for c in status.get("conditions", [])
            )
        ):
            continue
        fresh = copy.deepcopy(cfg)
        caps = {}
        for key, axis in FIELDS.items():
            total = quantity(status.get("allocatable", {}).get(key, 0))
            for pod in pods:
                if pod["spec"].get("nodeName") == meta["name"]:
                    total -= requests(pod["spec"], key)
                    if services and held_service(cfg, pod):
                        total += sum(
                            requests({"containers": [c]}, key)
                            for c in pod["spec"]["containers"]
                            if c["name"] in ("postgres", "minio")
                        )
            free = min(total, headroom.get(key, total))
            if key == "memory":
                free /= Decimal(2**30)
            ceiling = cfg["capacity"][axis] if axis in policy["declared_axes"] else BOUNDS[axis][1]
            caps[axis] = max(0, min(int(free), ceiling))
        if not roles.intersection(GPU_ROLES):
            caps["gpus"] = 0
        requested = dict(policy["requested"])
        if not roles.intersection(GPU_ROLES):
            requested.update(gpus=0, answer_tp=1)
        elif caps["gpus"] < 1:
            reasons.add("no available single-node GPU")
            continue
        if roles.intersection(GPU_ROLES) and requested["gpus"] == 0:
            raise Error("GPU phase cannot use a zero GPU ceiling")
        # Auto may reduce parallelism to fit CPU headroom or an explicit TP;
        # never alter the model, precision, context or image workload.
        counts = range(caps["gpus"], 0, -1) if requested["gpus"] == "auto" else [requested["gpus"]]
        for count in counts:
            fresh["capacity"] = dict(caps)
            fresh["resources"] = copy.deepcopy(policy["per_role"])
            try:
                normalize(fresh, {**requested, "gpus": count}, roles=roles)
                break
            except Error as exc:
                reasons.add(str(exc))
        else:
            continue
        fresh["resource_policy"] = {
            **policy,
            "source": "live_node_and_namespace",
            "node_hostname": host,
        }
        t = fresh["topology"]
        eligible.append(((-t["gpus"], -t["cpus"], -t["memory_gib"], host), fresh))
    check(
        eligible,
        "no eligible single node fits namespace and phase bounds: "
        + ", ".join(sorted(reasons) or ["no ready schedulable Linux node matched"]),
    )
    return min(eligible, key=lambda row: row[0])[1]


def resolve(cfg, invoke, roles=None):
    if cfg["cluster"]["discovery"] == "declared":
        check(cfg["resource_policy"]["explicit_envelope"], "explicit capacity envelope required")
        return cfg
    kube = ["kubectl", "--context", cfg["cluster"]["context"]]
    ns = cfg["cluster"]["namespace"]
    try:
        nodes = json.loads(invoke([*kube, "get", "nodes", "-o", "json"]))["items"]
        pods = projected_pods(
            invoke([*kube, "get", "pods", "--all-namespaces", "-o", pod_projection()])
        )
        quotas = json.loads(invoke([*kube, "-n", ns, "get", "resourcequota", "-o", "json"]))[
            "items"
        ]
    except (Error, KeyError, TypeError, ValueError) as exc:
        raise Error(
            "cannot inspect node allocations and namespace quota; grant read-only discovery "
            "or explicitly set discovery=declared with a verified capacity envelope"
        ) from exc
    try:
        return from_snapshot(cfg, nodes, pods, quotas, roles)
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise Error("invalid Kubernetes inventory or unsupported resource quantity") from exc
