# SSF

Strategy Search Framework for QSeaBattle.

## Documentation

Project documentation is published at: https://robhendrik.github.io/SSF/

## Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

- src/ssf/                 Core evaluators and utilities
- scripts/generate/       Fixture generators
- scripts/evaluate/       Bob-side evaluators
- scripts/optimize/       Parameter optimization/search
- scripts/compare/        Plotting and comparison tools
- fixtures/               Reusable reference fixtures
- results/                Generated experiment outputs
- tests/                  Pytest suite

## License

License: MIT. See LICENSE.
