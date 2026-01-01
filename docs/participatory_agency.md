# Participatory Agency Value Decomposition

The participatory agency package introduces a multi-head value decomposition layer for
GEPA models. The four heads are:

- **Epistemic humility**: awareness of uncertainty and calibration.
- **Cooperative equilibrium**: balancing outcomes in shared settings.
- **Goal flexibility**: adapting goals as context evolves.
- **Participatory identity / belonging**: groundedness in shared identity and inclusion.

These heads are optional and additive. They do not change any default training behavior
unless explicitly enabled.

## Minimal usage

```python
import torch

from gepa_mindfulness.participatory_agency import ParticipatoryValueHead

hidden_size = 768
value_head = ParticipatoryValueHead(hidden_size=hidden_size)

hidden_features = torch.randn(2, hidden_size)
values = value_head(hidden_features)
print(values.total())
```

## Optional integration example

```python
from gepa_mindfulness.integrations import attach_participatory_value_head

class MyModel:
    def __init__(self) -> None:
        self.hidden_size = 768

model = MyModel()
head = attach_participatory_value_head(model, hidden_size=model.hidden_size)
```

The integration helper attaches the value head without changing any GEPA training loops
unless you opt in to use it.
