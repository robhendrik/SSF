## Dense Fixture Contract for A-Side Strategy Tables

This section defines how an abstract hybrid strategy is encoded as dense `.npz` tables. It is the authoritative implementation contract for `hybrid_strategy_fixtures.py`.

### Stored Arrays

A dense A-side fixture contains:

n2: int
k_box: int
comm_table: uint8 array, shape (2**n2, 2**k_box)
f_table_0 ... f_table_{k_box-1}

For box index `d`:

f_table_d.shape == (2**n2, 2**d)

Rows are indexed by `field`, the integer encoding of Alice’s input bits. Columns are indexed by `out_a_prefix`, the integer encoding of Alice’s outputs from previous boxes:

out_a_prefix = out_a & ((1 << d) - 1)

Bit `r` of `out_a_prefix` is the A-side output of box `r`.

### Meaning of `f_table_d`

`f_table_d[field, out_a_prefix]` stores the INPUT bit sent by Alice into PR box `d`.

It does NOT store the post-box unit output.

For a pyramid unit combining values `p0` and `p1`:

a_d = p0 XOR p1
o_A_d = bit d of out_a
unit_output = p0 XOR o_A_d

The dense table stores:

f_table_d[field, out_a_prefix] = a_d

NOT:

unit_output

The `unit_output` is only used internally when computing later table values and the final communication bit.

### Meaning of `comm_table`

`comm_table[field, out_a]` stores Alice’s final communication bit `c` for the complete field and complete vector of A-side PR-box outputs.

comm_table.shape == (2**n2, 2**k_box)

For every row:

field ∈ [0, 2**n2)
out_a ∈ [0, 2**k_box)

`comm_table[field, out_a]` must recompute the full A-side strategy using the bits of `out_a` as the PR-box A-side outputs.

Unused boxes, if any, must not influence `comm_table`.

### Field Bit Ordering

The input field integer uses little-endian bit indexing:

x_i = (field >> i) & 1

So bit `0` of `field` is `x_0`, bit `1` is `x_1`, etc.

### A-Side Output Bit Ordering

The A-side PR-output integer also uses little-endian bit indexing:

o_A_d = (out_a >> d) & 1

So bit `0` of `out_a` is the output of PR box `0`, bit `1` is the output of PR box `1`, etc.

### Canonical Box Ordering for Pyramid Units

Pyramid boxes are assigned breadth-first, layer by layer, left to right.

For `n2 = 8`, the pure pyramid uses seven boxes:

Layer 0:
  box 0: combine x0, x1
  box 1: combine x2, x3
  box 2: combine x4, x5
  box 3: combine x6, x7

Layer 1:
  box 4: combine output(box 0), output(box 1)
  box 5: combine output(box 2), output(box 3)

Layer 2:
  box 6: combine output(box 4), output(box 5)

For every unit:

a_d = left_value XOR right_value
unit_output = left_value XOR o_A_d

The final apex value is the communication bit.

This same ordering must be used for:
- pure pyramid,
- horizontal pyramid branches,
- vertical pyr/maj partial pyramid layers,
- vertical maj/pyr pyramid over majority-block outputs.

### Horizontal Hybrid Encoding

For `family="horizontal"`:

order="maj_pyr"

means:
- majority is applied to the left prefix of length `split`;
- pyramid is applied to the right suffix of length `n2 - split`.

order="pyr_maj"

means:
- pyramid is applied to the left prefix of length `split`;
- majority is applied to the right suffix of length `n2 - split`.

The pyramid branch uses canonical pyramid box ordering over its local branch indices. Local branch bit `0` corresponds to the first bit in that branch.

If the majority side is not tied:

comm = majority result

If the majority side is tied:

comm = pyramid branch apex

### Vertical `pyr_maj` Encoding

For `family="vertical", order="pyr_maj"`:

1. Apply the first `depth_s` pyramid layers to the full input sequence.
2. The resulting apex sequence has length:

apex_len = n2 // 2**depth_s

3. Apply majority with `tie_bandwidth` to that apex sequence.
4. If the apex majority ties:

comm = tie_value

The PR boxes used are exactly the boxes in the first `depth_s` layers of the full canonical pyramid.

### Vertical `maj_pyr` Encoding

For `family="vertical", order="maj_pyr"`:

1. Partition the input into:

block_count = 2**depth_s
block_size = n2 // block_count

2. Apply majority with `tie_bandwidth` independently to each block.
3. If a block majority ties, that block output is:

tie_value

4. Apply a canonical pyramid over the `block_count` majority outputs.
5. The final pyramid apex is `comm`.

The PR boxes used are exactly:

block_count - 1

assigned in canonical pyramid order over the block outputs.

### Unused Boxes

For any valid spec:

required_boxes <= k_box

If:

required_boxes < k_box

then all boxes with index:

d >= required_boxes

are unused dummy boxes.

For every unused box:

f_table_d must be all zeros

Unused boxes must not influence `comm_table`.

### Invalid Configurations

The fixture builder must reject invalid configurations with `ValueError`.

Examples:
- not enough boxes: `required_boxes > k_box`;
- invalid split;
- majority side not even;
- pyramid side not power of two;
- invalid vertical depth;
- invalid or redundant tie bandwidth;
- unsupported family/order.

The builder must not silently repair invalid specs.

### Consistency Requirement

For any valid spec and any small feasible `n2`, dense fixtures must be equivalent to procedural evaluation:

f_table_d[field, out_a_prefix] == procedural.measure_a(d, field, out_a_prefix)

comm_table[field, out_a] == procedural.comm(field, out_a)

This equivalence should be tested exhaustively for small cases.