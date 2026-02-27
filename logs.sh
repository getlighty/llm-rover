#!/bin/bash
# View rover-brain service logs
# Usage: ./logs.sh        - tail live logs
#        ./logs.sh -a     - show all logs from current boot
#        ./logs.sh -n 50  - show last 50 lines

case "${1:-}" in
  -a|--all)
    journalctl -u rover-brain.service -b --no-pager
    ;;
  -n)
    journalctl -u rover-brain.service -b --no-pager -n "${2:-100}"
    ;;
  *)
    journalctl -u rover-brain.service -b -f
    ;;
esac
