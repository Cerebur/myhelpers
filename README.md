# myhelpers

Research helpers for working with spacecraft observations of Mercury and Venus. The package includes data loaders for magnetic field and particle instruments, Mercury crossing catalogues, MESSENGER data download utilities, and supporting plotting and SPICE tools.

## Install

From this directory, install the package and its dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Example

Load MESSENGER magnetic field data for a time interval:

```python
import pandas as pd
from myhelpers import load_mag_data

data, files = load_mag_data(
    time_interval=[
        pd.Timestamp("2011-03-23T15:39:00Z"),
        pd.Timestamp("2011-03-23T15:40:00Z"),
    ],
    spacecraft="messenger",
    coordinates="mso",
    data_local=True,
)
```

The data paths are configured in `src/myhelpers/parameters/paths.py`; update them to match your local data storage before using the loaders.
