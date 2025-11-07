import nbformat
from pathlib import Path

nb = nbformat.read(Path('equity_demo.ipynb'), as_version=4)
ns = {}
for idx, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    print(f'Executing cell {idx}')
    exec(cell.source, ns)
print('results keys:', list(ns.get('results', {}).keys()))
if ns.get('results'):
    payload = next(iter(ns['results'].values()))
    events = payload['events_equity']
    print('events_equity rows', len(events))
    print('events_equity head:\n', events.head())
