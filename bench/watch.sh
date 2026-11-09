#!/usr/bin/env bash
# Boot watchdog: log host RAM + container RSS + VRAM of GPUs 1,2 every 5 s; stop the
# container if available host RAM drops below LIMIT_G GiB.
LIMIT_G=${LIMIT_G:-10}
NAME=${NAME:-dsv41-flash-exl3}
LOG=${LOG:-/tmp/dsv41-watch.log}
: > "$LOG"
while true; do
  av=$(free -g | awk 'NR==2{print $7}')
  cm=$(docker stats --no-stream --format '{{.MemUsage}}' "$NAME" 2>/dev/null | cut -d/ -f1)
  vr=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 1,2 | tr '\n' ' ')
  echo "$(date +%T) avail=${av}G cont=${cm:-none} vram[idx MiB]=${vr}" >> "$LOG"
  if [ "$av" -lt "$LIMIT_G" ]; then
    echo "$(date +%T) LOW RAM (${av}G < ${LIMIT_G}G): stopping $NAME" >> "$LOG"
    docker stop -t 30 "$NAME" > /dev/null 2>&1
    exit 2
  fi
  docker ps --format '{{.Names}}' | grep -q "^${NAME}$" || { sleep 5; docker ps --format '{{.Names}}' | grep -q "^${NAME}$" || { echo "$(date +%T) container gone" >> "$LOG"; exit 0; }; }
  sleep 5
done
