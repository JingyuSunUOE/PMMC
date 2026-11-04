# Deploy on Kubernetes

The controller runs outside the experiment containers. It uses Python 3.11/3.12, kubectl and a persistent, operator-owned host state directory (`PMMC_DEPLOY_STATE`, default `~/.local/state/pmmc-release`). Use encrypted private storage for controller state, which includes service credentials. A lost controller state directory is not permission to adopt an existing service volume or regenerate its password.

## Cluster and storage

Start with the [generic template](../deploy/release/kubernetes.toml). Install the external [release image lock](release-images.md). Set the context and a namespace you own; the controller never creates or wipes the cluster. A cluster administrator must already have installed a dynamic provisioner and, for inference, compatible NVIDIA drivers and device plugin. Standard Jobs are unsuspended. Set `cluster.queue` **only** for a cluster which supports the required Kueue workload integrations.

```bash
kubectl config current-context
kubectl get storageclass
kubectl get pvc --namespace "$PMMC_NAMESPACE"
```

The database and object store each have a separate namespaced RWO claim, internal Service, StatefulSet, readiness probe and credential binding. Select the approved RWO storage class and sizes. On first launch the instance generates independent credentials, initializes roles/pgvector/migrations and restricted object-store access. Configuration conflicts stop the launcher; it must not empty PGDATA, rotate the instance credentials, force queue admission or replace a bound volume to resolve an error.

Use a **different** work claim for the disjoint `mounts.inputs`, `outputs`, `artifacts` and `cache` directories. For dynamic creation set `create_shared_pvc = true`, `shared_storage_class`, `shared_pvc` and `shared_storage_gib` in `[cluster]`. Otherwise provide an already-provisioned claim and the writable subdirectories. Use RWX for multi-node access. A single-node RWO work claim requires `shared_access_mode = "ReadWriteOnce"` and `node_hostname` set to the node where all its consumers can mount it. A StorageClass using WaitForFirstConsumer may remain Pending until a consuming pod is scheduled. Do not replace such a claim to make it bind.

Check that the assigned non-root UID/GID can create files in these dedicated directories; storage with root squash or site-specific ACLs may require administrator provisioning. Do not recursively chown shared storage. Bootstrap uses service-admin credentials; the scientific worker is not given the MinIO root password. Assets are staged only by the downloader, then mounted read-only for consumers. CPU analysis uses saved outputs, not a writable source copy. There is no public service ingress; secure internal traffic, Secret storage, registry access, resource quotas and external storage according to local policy.

## Resource envelope

`resources.gpus`, `cpus` and `memory_gib` are total ceilings. The CLI overrides those ceilings for a new run. Live discovery reads candidate nodes, active pod requests and namespace quota, deducts existing allocations, and selects a feasible **single-node** envelope. It is an admission estimate, not a GPU reservation. Discovery needs read access to Nodes and cluster-wide Pods. If RBAC forbids these reads, an operator may select `discovery = "declared"` with an explicit verified per-node capacity envelope. A missing permission is not reported as capacity.

Resource requests include service reservations and the CPU control worker when concurrent GPU work is planned. Tensor parallelism and replica counts must fit; GPU calibration and native token/image bounds still apply. Additional replicas do not imply identical performance or capacity across hardware. Fixed topology, model, precision or hardware changes cannot inherit an incompatible capacity measurement. For an optional queue, use the site's LocalQueue; watch Pending, admitted, initializing, running and terminal states separately. Never manually unsuspend a queued PMMC Job.

## First use, restart and diagnostics

`services up` owns only this instance's database/object-store resources. `prepare` materializes selected inputs, model dependencies, snapshots and capacity. `run` composes these operations. For source revisions, upstream terms and local-input mode see [benchmark preparation](release-benchmarks.md). Use the [README](../README.md) commands for main, analysis, standalone baseline, quote/approve and resume.

A run freezes its plan, resource configuration and workload names before submission. If the controller is interrupted, use `./scripts/pmmc resume --run ID`. It reconciles existing workload IDs; it does not resubmit a paid call with unknown outcome. Do not change an existing plan to hide failed positions. Use a new run for different selections or an incompatible deployment.

The bounded host archive printed as `SYNC_REVIEW_FILE` is private. It includes progress events, job conditions, safe failure codes, stage counts and a limited evidence projection, not full provider traces, raw data, credentials or a server backup. `./scripts/pmmc diagnostics --run ID --refresh` collects a new evidence projection. See [private delivery](private-deploy.md) for the separate, manually transferable encrypted deployment file.
