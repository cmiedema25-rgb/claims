#!/bin/bash
set -u
umask 022
mkdir -p /logs/verifier
chmod 755 /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
rm -f /logs/verifier/ctrf.json
set +e
cd /tests
python3 -m pytest -p no:cacheprovider --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
status=$?
if [ ! -f /logs/verifier/ctrf.json ]; then
  printf '{"results":[]}\n' > /logs/verifier/ctrf.json
fi
if [ "$status" -eq 0 ]; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
chmod 644 /logs/verifier/ctrf.json /logs/verifier/reward.txt
exit 0
