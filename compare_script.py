import numpy as np
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path("src")))

from ssf.hybrid_strategy_spec import HybridStrategySpec, validate_spec
from ssf.hybrid_strategy_fixtures import build_strategy_fixture

# Load the fixture
fixture_path = Path("fixtures/persistent/a_strategies/majority_n2_8_k_1.npz")
with np.load(fixture_path, allow_pickle=False) as data:
    actual_comm = np.asarray(data["comm_table"], dtype=np.uint8)

# Load manifest to see what the first entry says
with open("fixtures/persistent/a_strategies/manifest.json") as f:
    manifest = json.load(f)

# Find majority_n2_8_k_1 entries
entries = [e for e in manifest if e.get("file") == "majority_n2_8_k_1.npz"]
print(f"Found {len(entries)} manifest entries for majority_n2_8_k_1.npz")

for i, entry in enumerate(entries):
    print(f"\nEntry {i}: tie_bandwidth={entry.get('tie_bandwidth')}, tie_value={entry.get('tie_value')}")
    
    # Build expected fixture with this entry's tie_bandwidth
    spec = HybridStrategySpec(
        family="majority",
        n2=8,
        k_box=1,
        tie_bandwidth=entry.get('tie_bandwidth', 0),
        tie_value=entry.get('tie_value', 0),
    )
    spec = validate_spec(spec)
    expected = build_strategy_fixture(spec)
    expected_comm = expected.comm_table
    
    # Compare
    matches = np.array_equal(actual_comm, expected_comm)
    print(f"  Matches actual fixture: {matches}")
    
    if not matches:
        # Show some sample mismatches
        print(f"  Actual shape: {actual_comm.shape}, Expected shape: {expected_comm.shape}")
        diff_count = np.sum(actual_comm != expected_comm)
        print(f"  Number of differing elements: {diff_count}")
