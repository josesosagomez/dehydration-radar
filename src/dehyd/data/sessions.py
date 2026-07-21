"""The single definition of session order.

S0..S4 == 8am/10am/12pm/2pm/4pm == the paper's DeHydL0..L4. Nothing else belongs in
this module, and no config file duplicates this order — ground_truth.py,
loader_10ghz.py and manifest.py all import from here so they cannot drift apart.
"""

SESSION_NAMES = ("8am", "10am", "12pm", "2pm", "4pm")

SESSION_INDEX = {name: i for i, name in enumerate(SESSION_NAMES)}

N_SESSIONS = len(SESSION_NAMES)
