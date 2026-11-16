"""Author-only synthetic cohort provenance; not present in the agent image."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['claim_id', 'event_id', 'cohort_index', 'observation_age_days',
          'policy_limit_usd', 'closed', 'settlement_days', 'paid_indemnity_usd', 'limit_exhausted']


def observation(rng, cid, eid, period, event_effect, calendar_effect, age, limit):
    group = int(rng.random() >= .61)
    ms, mt, ss, st, rho = [(9.55, 3.65, .55, .50, -.42), (12.05, 5.00, .70, .62, .52)][group]
    z, w = rng.normal(size=2)
    severity = np.exp(ms+.46*event_effect+.57*calendar_effect+ss*z)
    duration = np.exp(mt+.51*event_effect-.34*calendar_effect+st*(rho*z+np.sqrt(1-rho*rho)*w))
    closed = duration <= age
    return dict(zip(FIELDS, [cid, eid, period, f'{age:.8f}', f'{limit:.8f}', int(closed),
                            f'{duration:.8f}' if closed else '', f'{min(severity,limit):.8f}' if closed else '',
                            int(severity >= limit) if closed else '']))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = np.random.default_rng(813629)
    market = rng.normal()
    history, current = [], []
    current_sizes = {25: (1, 4), 27: (1, 4), 29: (2, 3), 30: (2, 3)}
    current_ages = {25: [25,80,240], 27: [45,110,180], 29: [15,30,60], 30: [8,16,28]}
    for period in range(1, 31):
        if period > 1:
            market = .82*market+np.sqrt(1-.82**2)*rng.normal()
        event_count, current_size = (12, 0) if period <= 24 else current_sizes.get(period, (0, 0))
        for event in range(event_count):
            effect = rng.normal()
            prefix = 'H' if period <= 24 else 'C'
            eid = f'{prefix}-{period:02d}-{event+1:02d}'
            size = int(rng.integers(3, 7)) if period <= 24 else current_size
            for j in range(size):
                age = np.exp(rng.uniform(np.log(18),np.log(600))) if period <= 24 else float(rng.choice(current_ages[period]))
                limit = int(rng.choice([25000,60000,150000,400000,1000000])) if period <= 24 else int(rng.choice([90000,220000,400000,650000]))
                row = observation(rng,f'{eid}-{j+1}',eid,period,effect,market,age,limit)
                (history if period <= 24 else current).append(row)
    opens = [r for r in current if not r['closed']]
    events = sorted({r['event_id'] for r in opens})
    by_event = {e:[r for r in opens if r['event_id']==e] for e in events}
    pools = []

    def pool(pid, members, fraction):
        pools.append({'portfolio_id':pid,'claim_ids':[r['claim_id'] for r in members],
                      'attachment_usd':round(fraction*sum(float(r['policy_limit_usd']) for r in members)/25)*25})

    for event,members in by_event.items():
        if len(members)>=2:
            pool(event+'-OPEN',members,.4)
    for period in current_sizes:
        members = [r for r in opens if int(r['cohort_index'])==period]
        if members:
            pool(f'COHORT-{period}',members,.3)
    first = [by_event[e][0] for e in events]
    for offset in range(3):
        if first[offset::3]:
            pool(f'CROSS-{offset+1}',first[offset::3],.35)
    pool('ALL-CURRENT',opens,.3)
    data = ROOT/'environment/data'
    write_csv(data/'claim_history.csv',history)
    write_csv(data/'current_events.csv',current)
    queries = {'claims':[{'claim_id':r['claim_id']} for r in opens],'portfolios':pools}
    payload = json.dumps(queries,indent=2)+'\n'
    (data/'open_claims.json').write_text(payload)
    (ROOT/'tests/open_claims.json').write_text(payload)
    print(f'{len(history)} historical rows in 24 cohorts; {len(current)} current rows across cohorts 25..30 with two gaps; {len(opens)} open claims; {len(pools)} pools')


if __name__=='__main__':
    main()
