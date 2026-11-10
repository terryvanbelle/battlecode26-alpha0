#!/usr/bin/env bash
# Run one bounded, resumable campaign cycle in this checkout.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
campaign_hours=3
while (($#)); do
    case "$1" in
        --hours)
            [[ $# -ge 2 ]] || { echo '--hours requires a positive number' >&2; exit 2; }
            campaign_hours="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'HELP'
Usage: scripts/run_campaign.sh [--hours HOURS]

Run one resumable campaign cycle (default: 3 hours). Only one campaign may
run in this checkout. Logs and state are under runs/campaign/.

Create runs/campaign/STOP to request a graceful stop. A restart requires
explicitly removing that sentinel. The launcher never removes it for you.
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$campaign_hours" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
    [[ "$campaign_hours" =~ [1-9] ]] || {
        echo '--hours requires a positive number' >&2
        exit 2
    }

cd -- "$repo_dir"
campaign_dir="$repo_dir/runs/campaign"
mkdir -p -- "$campaign_dir"
command -v flock >/dev/null || { echo 'flock is required (util-linux)' >&2; exit 2; }
exec 9>"$campaign_dir/campaign.lock"
if ! flock -n 9; then
    echo 'Another campaign owns runs/campaign/campaign.lock; nothing started.' >&2
    exit 3
fi
if [[ -e "$campaign_dir/STOP" ]]; then
    echo 'runs/campaign/STOP exists; remove it only when an intentional restart is wanted.' >&2
    exit 4
fi
campaign_python="${ALPHA0_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$campaign_python" ]] || {
    echo 'Python environment is missing; run bash scripts/bootstrap.sh first.' >&2
    exit 2
}
[[ -f "$repo_dir/alpha0/campaign.py" ]] || {
    echo 'alpha0/campaign.py is missing; this checkout cannot run a campaign.' >&2
    exit 2
}
printf '%s\n' "$$" > "$campaign_dir/campaign.pid"
printf 'Campaign PID %s; log: %s\n' "$$" "$campaign_dir/campaign.log"
exec >>"$campaign_dir/campaign.log" 2>&1
printf '\n[%s] Starting bounded campaign, hours=%s, pid=%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$campaign_hours" "$$"
# The exec preserves fd 9, keeping the lock held for the campaign process.
exec "$campaign_python" -u -m alpha0.campaign --root . --hours "$campaign_hours"
