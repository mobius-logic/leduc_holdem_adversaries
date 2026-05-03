"""Quick smoke-test for server_ni: alternating turn order + 2-raise cap."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server_ni
from game.state import Round

# ---- Turn order alternation ----
s = server_ni.GameSession()
order = [s._agent_acts_first]
print(f"Hand 1  agent_acts_first={order[-1]}")

for i in range(4):
    while s.phase == "your_turn":
        s.apply_action("Fold")
    if s.phase == "hand_summary":
        s.start_next_hand()
        order.append(s._agent_acts_first)
        print(f"Hand {i+2}  agent_acts_first={order[-1]}")
    elif s.phase == "tournament_over":
        break

# Verify strict alternation
for i in range(1, len(order)):
    assert order[i] != order[i-1], f"Turn order did NOT alternate at hand {i+1}!"
print("Turn-order alternation: PASS\n")

# ---- 2-raise cap ----
s2 = server_ni.GameSession()
# Force raises_this_round to the cap and confirm Raise disappears
s2.state.raises_this_round = server_ni._PREFLOP_RAISE_CAP
legal = s2.state.get_legal_actions()
assert "Raise" not in legal, f"Raise still in legal after hitting cap: {legal}"
print(f"After cap ({server_ni._PREFLOP_RAISE_CAP} raises), legal={legal}")
print("2-raise-per-round cap:  PASS")
