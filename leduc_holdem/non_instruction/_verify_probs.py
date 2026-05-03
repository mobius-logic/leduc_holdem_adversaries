import sys, numpy as np
sys.path.insert(0, '.')
from non_instruction.server_ni import GameSession

s = GameSession('aggressive')
step = 0
while s.phase not in ('tournament_over',) and step < 200:
    d = s.to_dict()
    if s.phase == 'your_turn':
        s.apply_action(d['legal_actions'][0])
    elif s.phase == 'hand_summary':
        s.start_next_hand()
    step += 1

d = s.to_dict()
probs = d['lda_step_probs']
print(f'LDA entries: {len(probs)}')
for p in probs:
    total = p['analytical'] + p['conservative'] + p['aggressive'] + p['reckless']
    best = max((k for k in p if k != 'step'), key=lambda k: p[k])
    ana, con, agg, rec = p['analytical'], p['conservative'], p['aggressive'], p['reckless']
    print(f"  Step {p['step']:2d}: Ana={ana:.3f}  Con={con:.3f}  Agg={agg:.3f}  Rec={rec:.3f}  sum={total:.3f}  best={best}")
