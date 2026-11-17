# Shared by scripts/box_helper.sh and scripts/vm_helper.sh (source it).

# job_state DIR TAG prints the remote shell command that reports a request's state in sandbox DIR: rc=N once
# runs/box/TAG.rc exists, "RUNNING <last log line>" while a process with runs/box/TAG.log on its command line is
# alive, DIED otherwise. The command never spells the log path in one piece (the '' splits it), so pgrep -f cannot
# match the shell that runs the check: it did, and a request whose job died without an rc waited forever.
job_state() {
  echo "cd $1; if [ -f runs/box/$2.rc ]; then echo rc=\$(cat runs/box/$2.rc); elif pgrep -f 'box/$2[.]log' >/dev/null; then echo RUNNING \$(tail -1 runs/box/$2''.log | cut -c1-80); else echo DIED; fi"
}
