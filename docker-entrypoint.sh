#!/bin/bash
# ROCm: /dev/kfd + /dev/dri/renderD* thuộc group render/video của host với GID khác nhau tùy máy.
# group_add trong compose không giải quyết được vì GID không cố định (và base image không có group "render").
# Thay vào đó: container start as root, đọc GID thật của từng device, tạo group đúng GID đó cho user magician,
# rồi drop privileges chạy lệnh chính. Không phải AMD (không có device) thì bỏ qua, chạy luôn.
set -e
for dev in /dev/kfd /dev/dri/renderD*; do
  [ -e "$dev" ] || continue
  gid=$(stat -c %g "$dev")
  [ "$gid" = "0" ] && continue  # device của root: không cần group
  gname=$(getent group "$gid" | cut -d: -f1)
  if [ -z "$gname" ]; then
    gname="gpudev$gid"
    groupadd -g "$gid" "$gname" 2>/dev/null || true
  fi
  usermod -aG "$gname" magician 2>/dev/null || true
done
exec runuser -u magician -- "$@"
