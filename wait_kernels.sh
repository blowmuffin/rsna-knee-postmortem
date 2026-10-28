# usage: wait_kernels.sh k1 k2 ...  -> exits when any kernel reaches a terminal state, or after ~28 min
end=$(( $(date +%s) + 1500 ))
while [ $(date +%s) -lt $end ]; do
  for k in "$@"; do
    s=$(timeout 60 python -m kaggle kernels status ayushkasare/$k 2>&1)
    if echo "$s" | grep -qiE 'COMPLETE|ERROR|CANCEL'; then date -u; echo "DONE $k: $s"; exit 0; fi
  done
  sleep 90
done
date -u; for k in "$@"; do echo "still: $(timeout 60 python -m kaggle kernels status ayushkasare/$k 2>&1)"; done
