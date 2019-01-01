# AlphaGo Zero based RL agent 
Made during 'AI Weekend' in Stockholm.

## Structure
```python
├── README.md
├── agz.py              # MCTS logic. File can also visualise etc
├── goboard.py          # Go implementation code
├── scoring.py          # Go implementation code
├── gostate.py          # Go environment wrapping goboard, scoring
├── gostate_pachi.py    # Go environment wrapping the fast pachi implementation
├── resnet.py           # Neural network for evaluating board positions
├── policyvalue.py      # Predictor class wrapping the resnet CNN
└── training.py         # Training loop performing self play 
```

## Installation

Requires [pachi-py](https://github.com/openai/pachi-py).
```
pip install numpy
pip install keras
pip install tensorflow

python agz.py
```

## Todo
- [ ] Cleanup code structure with folders etc
- [ ] Implement random reflections of board
- [ ] Tune how much time is spend exploring / training (c.f. AGZ paper)
- [ ] Parallelize training and simulation.
- [ ] Use code from `agz.play_game` to create `MCTSAgent` class 
- [ ] Use same model on other environments
- [ ] Learn the transition dynamics of step(state, action)

(Refactor `MCTSAgent` to implement `.update_state` and `.decision` methods)
