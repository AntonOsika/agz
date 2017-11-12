# MCTS things 

## AlphaGo Zero hack during AI Weekend, Stockholm

python## Structure
```
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
- Implement `SimpleCNN` class that has `.predict` and `.train_on_batch` methods
- Use code from `agz.play_game` to create `MCTSAgent` class 
- Use same logic this on other environments
- Learn the transition dynamics of step(state, action)

`MCTSAgent` should have `.update_state` and `.decision` methods
