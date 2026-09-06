#!/usr/bin/env bash
# nextest run-wrapper: the one place tests coordinate GPU use with other
# processes. gpu tests share the lock, gpu-exclusive tests hold it alone, and
# CPU tests take nothing.
set -euo pipefail
readonly lock=$1
shift
case "${NEXTEST_TEST_GROUP:-}" in
  gpu) mode=--shared ;;
  gpu-exclusive) mode=--exclusive ;;
  *) exec "$@" ;;
esac
# An ancestor already holding the lock (`flock <lock> cargo nextest run ...`)
# passes its descriptor down; locking again would self-deadlock.
for fd in /proc/$$/fd/*; do
  if [[ ${fd} -ef ${lock} ]]; then exec "$@"; fi
done
# Tells a test that timed out waiting for another GPU user apart from a hang.
if ! flock --nonblock "${mode}" "${lock}" true; then
  echo "nextest_gpu_lock: waiting for ${mode#--} ${lock}" >&2
fi
exec flock "${mode}" "${lock}" "$@"
