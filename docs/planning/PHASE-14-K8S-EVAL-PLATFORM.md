# Phase 14 Implementation Plan: K8s Eval Platform — Portable SWE-bench Jobs on Ephemeral GKE Autopilot

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Draft v1.1 — implementation-ready (review fixes applied: emptyDir /tmp, prefix checkpoint sync, app label selector, Python GCS client, NetPol egress note)
**Parent Document:** `docs/planning/MASTER-PLAN.md`
**Dependencies:** Phase 5 complete (`evaluation/harness.py` + `evaluation/config.py` + `evaluation/test_runner.py`), Phase 6 complete (`inference/serve.py` + `inference/modal_serve.py`), Phase 1 complete (TF `storage`+`iam`, WIF, AR `docker_repo`), Phase 7 complete (`.github/workflows/{ci,cd,eval,promote}.yml`)
**Non-goals:** Moving prod serving off Modal. Standing GPU nodepool. Service mesh / GitOps operator.

---

## 1. Objective

Run SWE-bench eval as **portable, cheap, reproducible Kubernetes Jobs** while leaving proven Modal serving untouched:

- `evaluation/cli.py:run` → `EvaluationHarness.run_batch()` sharded by **repo** → one Indexed Job completion per repo shard → GCS checkpoints + results
- Ephemeral zonal GKE Autopilot (`us-central1-a`) — `terraform apply` to demo, `terraform destroy -target=module.gke` to $0
- TF owns platform, `kubectl apply -k` owns workloads (WikiStream v4 lesson)
- Hardened by default: WI, AR digest-pin + Trivy, ResourceQuota/LimitRange, NetworkPolicy default-deny, GMP PodMonitoring + log alert decoupled from eval F2P health
- Bounded GPU canary **last** (L4, port-forward only, 1hr teardown) for inference story — default is CPU-only

Resume line: *Portable SWE-bench eval on ephemeral GKE Autopilot (Indexed Jobs → GCS) with Modal A10G serving untouched; hardened TF + Kustomize + GMP; bounded KServe/vLLM canary.*

---

## 2. Inputs (verified in repo)

| Source | Artifact | Location |
| --- | --- | --- |
| Phase 5 | `EvaluationHarness.run_batch` per-repo checkpoint resume, `_run_tests_batch_modal` + fallback, `_run_tests_swebench` official images | `evaluation/harness.py:1185,341,367,421` |
| Phase 5 | `CheckpointManager` local `data/eval_checkpoints`, `WandbLogger`, `estimate_run_cost`, `latency_percentiles` | `evaluation/harness.py:531,650,925,906` |
| Phase 5 | `EvalConfig`: `golden_data_path=gs://swe-qwen-datasets/...`, `docker_image_base=python:3.11-slim`, `gpu_type=a10g-24gb`, `inference_gpu=a100-80gb`, `max_parallel=16`, `tier_sizes smoke20/dev100/final500/full50`, `modal_volumes repo_cache/test_cache`, `use_swebench_images=True` | `evaluation/config.py:20-110` |
| Phase 5 hotspot | `test_runner.py::run_tests_batch` cyclo 83, `_execute_instance` 68, `harness.run_batch` 64 — do NOT refactor, isolate via Job boundary | `get_repo_health` B/86.9 |
| Phase 6 | `create_app(engine, config)` POST `/v1/chat/completions` + GET `/health`, `VLLMEngine` lazy Linux-only + `StubEngine`, `ServeConfig` (`serving_hf_id=Qwen/Qwen3-14B-AWQ`, `gpu a10g-24gb`, `modal_volume serve-model-cache`) | `inference/serve.py:350,86,169`, `inference/config.py:40` |
| Phase 6 | `ModelServer` A10G:1 via `gpu_spec_for_tier`/`resolve_serving_gpu`, `concurrent_inputs=16`, scale-to-zero | `inference/modal_serve.py:183,45,56` |
| Phase 1 | TF `module storage` + `module iam` (GCS, WIF GH pool, SAs `modal_runner/github_actions/cloud_build`, Secrets modal/wandb/github, AR `docker_repo`) — **no GKE yet** | `infra/terraform/main.tf`, `modules/iam/main.tf` |
| Phase 7 | `cd.yml` TF+modal deploy, `ci.yml` lint/test+TF validate, `eval.yml` smoke gate, `promote.yml` champion flow | `.github/workflows/` |
| Phase 8 | `assert_registered`, per-request telemetry hooks | `observability/metrics.py`, `inference/serve.py::_record_and_trace` |
| Signals | `POST /v1/chat/completions` reach 132/38 files; `cli:run` reach 227 — eval CLI is orchestrator, serving is leaf | `get_signal_chains` |

---

## 3. Decisions Locked (grilled)

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| D1 | K8s scope | **Eval Jobs only** | Test shards stateless + checkpointed = best Job fit; serving already scale-to-zero A10G |
| D2 | GPU | **CPU default, ephemeral L4 demo bounded + torn down** (no standing pool) | Standing GPU = idle $$$ + recruiter red flag; 1hr demo proves GPU scheduling without cost bleed |
| D3 | Cluster | **Ephemeral zonal Autopilot `us-central1-a`** | No node mgmt, per-project, matches buckets region; `destroy` to $0 |
| D4 | Split | **TF platform, Kustomize workloads** | `infra/terraform/modules/gke/` vs `k8s/` — platform vs app lifecycle |
| D5 | Shard key | **repo** (one Indexed completion per repo) | Matches `run_batch` + `CheckpointManager.get_checkpoint_key(run_id,repo,model,variant,template)` |
| D6 | Checkpoint | **GCS prefix `gs://<dataset>/checkpoints/{run_id}/` (`*.json` per key `repo__model__variant__template`), full-prefix sync in worker** | No core `CheckpointManager` rewrite (local Path stays); worker DL/UL whole prefix |
| D7 | Image | **AR `docker_repo`, digest-pinned `python:3.11-slim`, Trivy gate** | Matches `EvalConfig.docker_image_base`, supply-chain story |
| D8 | Hardening | **WI KSA→GSA, Quota/LimitRange, NetPol deny-all, non-root + seccomp** | Standard-hardened without mesh/operator overbuild |
| D9 | Observability | **GMP PodMonitoring + log alert `EvalJobFailed`, decoupled from F2P/W&B health** | Infra alert ≠ eval quality gate |
| D10 | Evidence | **Repo artifacts (diagram + logs + GCS links + COST.md + RUNBOOK.md), no live URL** | Reproducibility > uptime for batch |

Rejected: full serving migration (loses scale-to-zero), standing GPU pool ($$$), Istio/Argo (overkill single-tenant batch).

---

## 4. Module Structure (new files only, flat)

```
infra/docker/eval-cpu.Dockerfile
evaluation/k8s_worker.py
scripts/make_eval_jobs.py
infra/terraform/modules/gke/{main.tf,variables.tf,outputs.tf}
infra/terraform/modules/iam/eval_runner.tf  # add-on, or extend main.tf
k8s/base/{namespace.yaml,serviceaccount.yaml,configmap.yaml,job.yaml,resourcequota.yaml,limitrange.yaml,networkpolicy.yaml,podmonitoring.yaml,kustomization.yaml}
k8s/overlays/{eval-smoke,eval-dev,eval-final}/kustomization.yaml
k8s/overlays/gpu-demo/{inferenceservice.yaml|deployment.yaml,kustomization.yaml}
.github/workflows/k8s-eval.yml
docs/k8s-eval.md  # thin pointer to this plan + run links
COST.md
RUNBOOK.md  # root or docs/, with make targets
```

No changes to `harness.py`, `test_runner.py`, `serve.py`, `modal_serve.py` except via `k8s_worker.py` wrapper.

---

## 5. Work Breakdown

### 5.1 P0 — Eval worker image + wrapper (start here)

**`infra/docker/eval-cpu.Dockerfile`**

```dockerfile
FROM python:3.11-slim@sha256:<pin-to-match-EvalConfig.docker_image_base>
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
COPY evaluation/ ./evaluation/
COPY registry/ ./registry/
COPY data_engineering/schema.py ./data_engineering/schema.py
USER 65532:65532
ENTRYPOINT ["python","-m","evaluation.k8s_worker"]
```

**`evaluation/k8s_worker.py` (~90 lines, spec)**

- argparse: `--run-id --repo --tier {smoke,dev,final} --golden-gcs --checkpoint-gcs --output-gcs`
- GCS via Python `google-cloud-storage` client (already a transitive dep through `_read_gcs`; do NOT install `gcloud` CLI in the slim image — CLI stays in CI only). On start: download whole prefix `{checkpoint-gcs}/{run_id}/*.json` → `/tmp/ckpt/{run_id}/` (keys are `{repo}__{model}__{variant}__{template}.json`, miss = fresh run) + load golden shard, filter `record.repo==args.repo`
- Build `EvalConfig(golden_data_path=args.golden_gcs, checkpoint_dir=Path("/tmp/ckpt"), output_dir=Path("/tmp/out"), dataset_run_id=args.run_id)`; `EvaluationHarness(config).run_batch(examples_for_repo, model, variant, template, run_id)`
- On return: upload `/tmp/out/*.jsonl` → `{output-gcs}/{run_id}/{repo}/` + `/tmp/ckpt/{run_id}/*.json` → `{checkpoint-gcs}/{run_id}/` (first write with `if_generation_match=0`; re-run skips completed repos via `is_completed`)
- Call existing `estimate_run_cost(results)` + print JSON summary to stdout (GMP logs scrape)
- Exit non-zero on exception → Job retry via `backoffLimit`

Verify: `docker build -f infra/docker/eval-cpu.Dockerfile -t eval-cpu:local . && trivy image --severity HIGH,CRITICAL --exit-code 1 eval-cpu:local`

### 5.2 P1 — TF platform `modules/gke`

**`infra/terraform/modules/gke/main.tf`**

```hcl
resource "google_container_cluster" "eval" {
  name     = "swe-qwen-eval-${var.environment}"
  location = var.zone  # default us-central1-a
  enable_autopilot = true
  deletion_protection = false
  release_channel { channel = "REGULAR" }
  workload_identity_config { workload_pool = "${var.project_id}.svc.id.goog" }
  resource_labels = var.labels
}
```

**`variables.tf`**: `project_id, region, zone=default us-central1-a, environment, labels`. **`outputs.tf`**: `cluster_name, endpoint`.

**IAM add-on (`modules/iam/eval_runner.tf`)**

- `google_service_account.eval_runner` + `google_service_account_iam_binding` WIF: `serviceAccount:${project}.svc.id.goog[swe-qwen-eval/eval-runner-ksa]`
- `google_project_iam_member`: `roles/storage.objectAdmin` (condition: `resource.name.startsWith("projects/_/buckets/${dataset}/objects/checkpoints/")` + results prefix), `roles/logging.logWriter`, `roles/monitoring.metricWriter`, `roles/artifactregistry.reader`

Wire in `infra/terraform/main.tf`: `module "gke" { source="./modules/gke" ... }`.

Commands:

```
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform plan -out=tfplan
terraform -chdir=infra/terraform apply tfplan
gcloud container clusters get-credentials swe-qwen-eval-dev --zone us-central1-a
```

### 5.3 P2 — Kustomize workloads

**`k8s/base/namespace.yaml`**: `apiVersion: v1, kind: Namespace, metadata: {name: swe-qwen-eval}`.
**`serviceaccount.yaml`**: `kind: ServiceAccount, metadata: {name: eval-runner-ksa, namespace: swe-qwen-eval, annotations: {iam.gke.io/gcp-service-account: eval-runner@<project>.iam.gserviceaccount.com}}`.
**`configmap.yaml`**: `GOLDEN_GCS`, `CHECKPOINT_GCS`, `OUTPUT_GCS`, `TIER`, `PYTHONUNBUFFERED=1`.
**`job.yaml`** (template, `scripts/make_eval_jobs.py` stamps `completions=N`):

```yaml
apiVersion: batch/v1
kind: Job
metadata: {name: eval-{run_id}, namespace: swe-qwen-eval}
spec:
  completionMode: Indexed
  completions: 12  # = repo shard count
  parallelism: 8
  backoffLimit: 2
  activeDeadlineSeconds: 3600
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {app: eval-worker}  # stable selector: label globs (job-name: eval-*) are invalid in selectors
    spec:
      serviceAccountName: eval-runner-ksa
      restartPolicy: Never
      securityContext: {runAsNonRoot: true, seccompProfile: {type: RuntimeDefault}}
      volumes:
      - {name: tmp, emptyDir: {}}  # required: readOnlyRootFilesystem + worker writes /tmp/ckpt + /tmp/out
      containers:
      - name: eval-worker
        image: <region>-docker.pkg.dev/<project>/docker-repo/eval-cpu@sha256:<digest>
        imagePullPolicy: IfNotPresent
        securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: {drop: ["ALL"]}}
        resources: {requests: {cpu: 500m, memory: 1Gi}, limits: {cpu: "2", memory: 4Gi}}
        volumeMounts:
        - {name: tmp, mountPath: /tmp}
        env:
        - {name: RUN_ID, value: "20260910-dev"}
        - {name: SHARD_INDEX, valueFrom: {fieldRef: {fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']}}}
        - {name: GOLDEN_GCS, valueFrom: {configMapKeyRef: {name: eval-config, key: GOLDEN_GCS}}}
        command: ["python","-m","evaluation.k8s_worker","--run-id","$(RUN_ID)","--shard-index","$(SHARD_INDEX)"]
```

**`resourcequota.yaml`**: `hard: {cpu: "16", memory: 32Gi, pods: "20", count/jobs.batch: "5"}`.
**`limitrange.yaml`**: default `cpu 500m/mem 1Gi`, max `cpu 2/mem 4Gi`.
**`networkpolicy.yaml`**: default-deny ingress+egress + allow `kube-dns:53`, egress `storage.googleapis.com:443`, `wandb.ai:443`. AR egress (`*.pkg.dev:443`) is NOT expressible in vanilla K8s NetPol (no DNS-wildcard egress) — v1 allows `0.0.0.0/0:443` with a comment; graduate to GKE FQDNNetworkPolicy when multi-env needs it.
**`kustomization.yaml`**: list all bases + `images:` replaced per overlay with digest.

**`scripts/make_eval_jobs.py`**: reads golden manifest (or `config` tier size), writes `k8s/overlays/eval-{tier}/kustomization.yaml` patches (`completions`, `parallelism`, `RUN_ID`, image digest). Overlays: `eval-smoke` (completions=2 for smoke20, parallelism 2), `eval-dev` (10 repos), `eval-final` (full).

Apply: `kustomize build k8s/overlays/eval-smoke | kubectl apply -f - && kubectl wait --for=condition=complete job/eval-<run_id> -n swe-qwen-eval --timeout=60m`

### 5.4 P3 — GPU canary (bounded, LAST)

**`k8s/overlays/gpu-demo/`**: prefer raw `Deployment+Service` (no KServe install = fewer moving parts) unless KServe story wanted for CV:

```yaml
# deployment.yaml (spec)
nodeSelector: {cloud.google.com/gke-accelerator: nvidia-l4}
tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
containers:
- name: serve-demo
  image: <AR>/serve-gpu@sha256:<digest>  # debian_slim + vllm>=0.26, SERVING_STUB=0
  ports: [{containerPort: 8000}]
  readinessProbe: {httpGet: {path: /health, port: 8000}, periodSeconds: 10}
  resources: {limits: {nvidia.com/gpu: 1, memory: 16Gi, cpu: "4"}}
```

No Ingress. Verify: `kubectl port-forward svc/serve-demo 8000:80 -n swe-qwen-eval & curl localhost:8000/health; curl -H "Authorization: Bearer $SERVE_TOKEN" localhost:8000/v1/chat/completions -d '{"model":"qwen3-14b:higher_rank_14b","messages":[{"role":"user","content":"fix: hello"}]}'`. Teardown same hour: `kubectl delete -k k8s/overlays/gpu-demo`.

### 5.5 P4 — Observability

**`podmonitoring.yaml`** (GMP):

```yaml
apiVersion: monitoring.googleapis.com/v1
kind: PodMonitoring
metadata: {name: eval-worker, namespace: swe-qwen-eval}
spec: {selector: {matchLabels: {app: eval-worker}}, endpoints: [{port: 8000, path: /metrics, interval: 30s}]}
```

If no `/metrics` on worker, rely on stdout JSON + log-based metrics (simpler, keep PodMonitoring minimal or drop port and use `kubectl logs` + W&B).

Log alert `EvalJobFailed`: filter `resource.type="k8s_container" AND resource.labels.namespace_name="swe-qwen-eval" AND (textPayload=~"BackoffLimitExceeded" OR jsonPayload.reason="Failed")`, alignment 5m, notify email. Document: infra alert ≠ `F2P/P2P` quality gate (`evaluation/metrics.py`, W&B).

### 5.6 P5 — CI `.github/workflows/k8s-eval.yml`

`on: workflow_dispatch (inputs: tier default smoke, run_id, teardown bool default true)`.
Jobs (each `ubuntu-latest`, WIF via existing pool):

1. `build-push`: `docker buildx build --push` to AR with `${{ github.sha }}` tag, `echo digest=$(...) >> $GITHUB_OUTPUT`, `trivy image --exit-code 1`.
2. `tf-apply`: `terraform init/plan/apply -target=module.gke` (idempotent if cluster exists).
3. `deploy-eval`: `gcloud get-credentials`, `python scripts/make_eval_jobs.py --tier ${{ inputs.tier }} --digest ${{ needs.build-push.outputs.digest }}`, `kustomize build | kubectl apply -f -`, `kubectl wait --for=condition=complete`.
4. `collect`: `kubectl logs -l job-name --prefix > logs/k8s-eval.log`, `gcloud storage ls $OUTPUT_GCS/$RUN_ID/`, append W&B run link + `estimate_run_cost` to `$GITHUB_STEP_SUMMARY`.
5. `destroy` (if teardown): `terraform destroy -target=module.gke -auto-approve`.

Add to `ci.yml`: `kustomize build k8s/overlays/eval-smoke | kubeconform -strict`.

Keep `eval.yml` Modal smoke as fast gate; K8s for `dev/final` + promotion challenger runs.

---

## 6. Cost (delta vs today, us-central1)

| Item | Cost | Note |
| --- | --- | --- |
| Autopilot control plane | $0 (Autopilot) / ~$74/mo if Standard — use Autopilot | Ephemeral, destroy when idle |
| CPU Jobs smoke20 (2 pods × ~15m × 0.5-2 vCPU) | ~$0.05-0.20/run | `activeDeadlineSeconds` caps runaway |
| GCS checkpoints/results | <$1/mo | Lifecycle: delete `checkpoints/*` after 30d |
| AR storage + egress | <$1/mo | Digest GC policy |
| GMP metrics/logs | free tier | 5m alert, no custom SLO burn |
| L4 demo 1hr | ~$0.70-0.90 | Torn down same session, logged in PR |
| **Idle delta** | **$0** (destroyed) | Document `terraform destroy` in RUNBOOK |

Standing A10G pool rejected: ~$1.2-1.8/hr idle = ~$900+/mo for portfolio with zero traffic.

---

## 7. Security

- WIF only, no JSON keys. KSA `eval-runner-ksa` → GSA `eval-runner` (least-privilege + bucket-prefix conditions).
- AR digest (`@sha256:`) in Job, `imagePullPolicy: IfNotPresent`, Trivy HIGH/CRITICAL gate in CI.
- `runAsNonRoot:65532`, `readOnlyRootFilesystem:true`, `seccomp RuntimeDefault`, `drop ALL`.
- NetPol deny-all default; egress allowlist DNS + GCS + W&B on 443, AR via `0.0.0.0/0:443` in v1 (vanilla NetPol can't wildcard `*.pkg.dev`; FQDNNetworkPolicy later).
- Secrets (`SERVE_TOKEN`, `WANDB_API_KEY`) via Secret Manager → env, never in ConfigMap/logs.

---

## 8. Hiring Evidence (repo artifacts)

- `docs/k8s-eval.md`: 1 diagram + D1-D10 table + `kubectl get jobs` + GCS/W&B links.
- `COST.md`: table §6 + one real `estimate_run_cost` JSON from smoke run.
- `RUNBOOK.md`: `make k8s-up / k8s-eval TIER=smoke RUN_ID=... / k8s-logs / k8s-down` wrapping TF + `make_eval_jobs` + kubectl.
- PR template: image digest, Job `Complete N/N`, `gcloud storage ls` output, W&B eval-aggregate link, destroy confirmation.

Interview deflects: *Why Jobs not Deployments?* (batch + retry + Indexed completions map to repo shards + TTL). *Why Autopilot zonal ephemeral?* (no node ops, cheapest demo, matches single-tenant batch). *Why not serve on K8s?* (Modal scale-to-zero A10G proven; K8s GPU would idle). *Checkpoint idempotency?* (`get_checkpoint_key` stable + GCS `if_generation_match:0` on first write, re-run skips completed repos).

---

## 9. Risks

- Hotspots `run_tests_batch:83` flaky under parallel → Job `backoffLimit:2` + per-repo isolation contains blast radius; no logic edit in v1.
- `use_swebench_images=True` pulls large per-repo images on Autopilot → set `parallelism ≤8`, document pull time in smoke report; fallback `_run_tests_batch_fallback` stays.
- SWE-bench image pull secrets / rate limits → prefer `docker.io` mirror via AR remote, note in runbook.
- 2 import cycles + 12 unstable modules → worker image is self-contained copy, doesn't fix cycles (out of scope, tracked).

---

## 10. Definition of Done

- [ ] `kustomize build k8s/overlays/eval-smoke` + `kubeconform -strict` pass in `ci.yml`
- [ ] Trivy 0 HIGH/CRITICAL on `eval-cpu`
- [ ] `eval-<run_id>` Job `Complete` (completions == repo shards) on smoke20, `kubectl logs` archived
- [ ] `gs://.../checkpoints/{run_id}/` + `gs://.../eval_results/{run_id}/` present, re-run skips completed (resume proven)
- [ ] W&B `eval-aggregate-{run_id}` linked in PR summary + `estimate_run_cost` logged
- [ ] `EvalJobFailed` alert fires on forced failure (test with `backoffLimit:0` canary), no coupling to F2P gate
- [ ] `terraform destroy -target=module.gke` → $0 compute, `COST.md` updated with real numbers
- [ ] `RUNBOOK.md` + `docs/k8s-eval.md` merged; GPU demo teardown logged (if run)

---

## 11. Build Order (for implementer)

1. P0 Dockerfile + `k8s_worker.py` + local `docker build` + Trivy
2. P1 `modules/gke` + `eval_runner` IAM + `terraform plan`
3. P2 base manifests + `make_eval_jobs.py` + `eval-smoke` overlay + `kubeconform`
4. P5 `k8s-eval.yml` (build → apply → wait → collect → destroy) on smoke20
5. P4 PodMonitoring + `EvalJobFailed` alert + COST/RUNBOOK docs
6. P3 GPU demo LAST, 1hr timebox, teardown + cost log

// ponytail: ephemeral zonal Autopilot + Job-per-repo is the smallest honest K8s that still reads senior; mesh/GitOps/standing-GPU when multi-team needs it.
