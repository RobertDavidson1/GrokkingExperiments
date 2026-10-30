
### Getting started

```Bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Train the model
grok train
```

### File Structure
```
├── src
│   └── grokkingexperiments
│       ├── cli.py      # CLI commands
│       ├── data.py     # Converts data to tensor and build train/test sets
│       ├── groups.py   # Data generation
│       ├── main.py     # Inits model, get splits, train and plot
│       ├── model.py    # Defines the Grokking Model
│       ├── train.py    # Defines the training loop
│       ├── utils.py    # Utilities, plotting metrics
│       └── notebooks   # Notebooks for dev / exploration
│           ├── DevModel.ipynb             # Initial Model Work
│           ├── SimpleTrainInterface.ipynb # Demonstrates simpler training
│           └── SymmetricGroup.ipynb       # Work on grokking Sym. Group
```


