# MCTS things 

## AlphaGo Zero hack during AI Weekend, Stockholm

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
pip install tqdm

python agz.py
```

## Todo
- Implement random reflections of board (mcts kind of ruins things now)
- Tune how much time is spend exploring / training (c.f. AGZ paper)
- Parallelize training and simulation.
- Use code from `agz.play_game` to create `MCTSAgent` class 
- Use same logic this on other environments
- Learn the transition dynamics of step(state, action)

(`MCTSAgent` should probably implement `.update_state` and `.decision` methods)
