# usage: wait_sub.sh ref1 [ref2 ...] -> exits when any listed submission is no longer PENDING, or after ~25 min
end=$(( $(date +%s) + 1500 ))
while [ $(date +%s) -lt $end ]; do
  all=$(cd /d/kaggle/rsna-knee && timeout 60 python -m kaggle competitions submissions -c rsna-knee-abnormality-detection 2>&1)
  for r in "$@"; do
    s=$(echo "$all" | grep "^$r ")
    if [ -n "$s" ] && ! echo "$s" | grep -q PENDING; then date -u; echo "DONE: $s"; exit 0; fi
  done
  sleep 180
done
date -u; echo "still pending: $*"
