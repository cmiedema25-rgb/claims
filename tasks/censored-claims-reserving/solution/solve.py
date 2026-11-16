#!/usr/bin/env python3
import json
import os
from pathlib import Path
from reserving import load_history, calibrate, predict

if __name__ == '__main__':
    app = Path(os.environ.get('APP_ROOT', '/app'))
    history = load_history(app/'data/claim_history.csv')
    fitted = calibrate(history)
    current = load_history(app/'data/current_events.csv')
    queries = json.loads((app/'data/open_claims.json').read_text())
    result = predict(fitted, queries, current, history)
    (app/'reserves.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Calibrated parameter vector:', fitted, flush=True)
