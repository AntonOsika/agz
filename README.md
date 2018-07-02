# AlphaGo Zero based RL agent 
Made during 'AI Weekend' in Stockholm.

## Structure
```python
├── README.md
├── agz.py          # MCTS logic. File can be run for visualisations etc
├── gostate.py      # Go environment
├── policyvalue.py  # Neural networks etc for evaluating board positions
├── goboard.py      # Go code 
├── scoring.py      # More go code 
└── training.py     # Training loop performing self play 
```

## Installation

```
pip install numpy
pip install keras
pip install tensorflow

python agz.py
```

## Todo
[ ] Implement random reflections of board
[ ] Tune how much time is spend exploring / training (c.f. AGZ paper)
[ ] Parallelize training and simulation.
[ ] Use code from `agz.play_game` to create `MCTSAgent` class 
[ ] Use same model on other environments
[ ] Learn the transition dynamics of step(state, action)

(Refactor `MCTSAgent` to implement `.update_state` and `.decision` methods)
