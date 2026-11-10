# Local and GCloud compute

The initial experiment uses the attached CPU executor. No GPU was detected. At setup it exposed 5 logical CPUs with a cgroup quota of 4 CPU cores and a 16 GiB cgroup memory limit. The training and official-engine worker processes run locally; there is no connection to the Battlecode submission service.

## Existing `battlecode-dev` VM

`terryvanbelle/battlecode26b-vibe` at commit `f4d6ec971b13a20f7f38fb56d249f62c0251ed05`, `tools/vm.sh`, records:

| Setting | Recorded value |
| --- | --- |
| Project | `tvanbelle-vibecode` |
| Zone | `us-west1-b` |
| Instance | `battlecode-dev` |
| Machine type | `e2-standard-8` (8 vCPUs, about 31 GB RAM) |
| Internal address | `10.138.0.3` |
| Original driver | `claude-driver` |
| Original SSH key path | `~/.ssh/google_compute_engine` |
| Existing Battlecode workspace | `~/projects/vibe/2026b` |

These are historical connection instructions, not a live resource measurement. The VM also hosts an existing private Battlecode ladder, so new training should use a separate directory and respect its other workloads.

The current executor does **not** have the original SSH key, a forwarded SSH agent, a GCloud configuration, Application Default Credentials, or a configured GCP outbound identity. Searching the source repository found references to credential paths, but no embedded private keys, service-account JSON, refresh-token JSON, or GitHub personal tokens. The paths to the private ladder's passwords and account files are also absent here; those credentials would not by themselves grant VM SSH access.

To run there from an already authenticated GCloud machine, first inspect resources and existing workloads:

```bash
gcloud compute instances describe battlecode-dev \
  --project=tvanbelle-vibecode --zone=us-west1-b \
  --format='yaml(status,machineType,networkInterfaces)'
gcloud compute ssh battlecode-dev \
  --project=tvanbelle-vibecode --zone=us-west1-b \
  --command='nproc; free -h; df -h; ps -eo pid,pcpu,pmem,args --sort=-pcpu | head -20'
```

Use the repository's normal bootstrap and training commands in a new checkout, for example `~/projects/battlecode26-alpha0`. Start with a small number of engine workers and leave capacity for the ladder. Preserve SSH host-key verification. Do not copy credentials into this repository or reuse the old script's disabled host-key checks.

Access can be enabled by configuring a GCP identity or approved SSH access in the managed environment settings. The private address also requires a route to that VPC, or an authorized GCloud IAP/external-address path. Merely installing the GCloud CLI does not supply either authentication or network reachability.

## GitHub access

At setup, the attached GitHub integration authenticated as `terryvanbelle`, but both GraphQL repository creation and REST `POST /user/repos` returned “Resource not accessible by integration.” The owner then created the public `terryvanbelle/battlecode26-alpha0` repository, and the local checkout was connected to it. No alternate GitHub creation credential was found in the source repository.
