# Hybrid Majority–Pyramid Algorithms
### Final Reference Specification: Verbal Description and Formal Analysis

---

## 0. Notation and Conventions

Throughout this document the following notation is used consistently:

| Symbol | Meaning |
|---|---|
| $L$ | Field length (number of bits in A's input sequence) |
| $x \in \{0,1\}^L$ | Input bit sequence at A-side. We use **"field"** and **"input sequence"** interchangeably. |
| $j \in \{0,\ldots,L-1\}$ | Index queried at B-side |
| $x_j$ | Target bit: the value B is trying to guess |
| $c \in \{0,1\}$ | Communication bit sent by A |
| $\hat{x}_j \in \{0,1\}$ | B's guess for $x_j$ |
| $f_B(j) \in \{0,1\}$ | B's **flip function**. We use **"flip"** and **"conditional flip"** interchangeably. |
| $b$ | Tie bandwidth parameter for majority |
| $s$ | Depth parameter for vertical hybrids |
| $p_{\text{rule}}$ | PR-box correlation quality ($0 \leq p_{\text{rule}} \leq 1$) |
| $k$ | $\log_2 L$ (pyramid depth, used when $L = 2^k$) |

B's guess is always formed as:

$$\hat{x}_j = c \oplus f_B(j)$$

Success means $\hat{x}_j = x_j$.

**Tie handling is strategy-specific** and part of the algorithm definition; there is no global rule across all strategies. Each strategy section explicitly specifies what happens on a tie.

---

## 1. Problem Setting

Two parties, A and B, share a resource (PR boxes, described below) and operate without a classical communication channel during execution — except for a single bit $c$ that A sends to B.

- **A** receives the full input sequence $x \in \{0,1\}^L$ and sends a single communication bit $c \in \{0,1\}$.
- **B** receives only an index $j \in \{0,\ldots,L-1\}$ and must produce a guess $\hat{x}_j$ for the value of $x_j$.

The goal is to design A's encoding strategy and B's decoding strategy — using shared PR-box correlations — to maximise the probability that $\hat{x}_j = x_j$, assuming $x$ is drawn uniformly at random from $\{0,1\}^L$.

---

## 2. PR-Box Model

### 2.1 Description

A **Popescu–Rohrlich (PR) box** is a hypothetical non-local device shared between two parties. It consists of a correlated pair: a box on A's side and a corresponding box on B's side. Each side accepts a binary input and produces a binary output.

For a pair with inputs $a$ (A-side) and $b$ (B-side), and outputs $o_A$ and $o_B$:

- With probability $p_{\text{rule}}$: the outputs satisfy $o_A \oplus o_B = a \land b$
- With probability $1 - p_{\text{rule}}$: the outputs satisfy $o_A \oplus o_B = 1 \oplus (a \land b)$

In words: B always produces the same output as A, except when **both** inputs are 1 — in that case the outputs differ with probability $p_{\text{rule}}$ (following the rule). When $p_{\text{rule}} = 1$ this is the ideal PR box. When $p_{\text{rule}} = 1/2$ the boxes are uncorrelated.

**Marginals:** For any individual box, the output is uniformly random: $o_A \sim \text{Uniform}\{0,1\}$, $o_B \sim \text{Uniform}\{0,1\}$. The correlation only becomes visible when both sides' outputs are compared.

All PR-box pairs are independent of each other and of the input $x$.

### 2.2 Indexing and Execution Model

PR boxes are indexed by a layer $\ell$ and a position $i$ within that layer: $\text{PR}(\ell, i)$.

**Execution model:** Each $\text{PR}(\ell, i)$ instance is used exactly once per protocol execution and produces one correlated output pair $(o_A, o_B)$. A and B share the same physical instance: when A calls $\text{PR}_A(\ell, i, a)$ and B calls $\text{PR}_B(\ell, i, b)$, the outputs $o_A$ and $o_B$ are jointly drawn according to the PR-box rule above. The randomness is shared between A and B through the box; neither party can influence the other's output directly.

---

## 3. Majority Operator

### 3.1 Verbal Description

The majority operator takes an input sequence of **even length** $n$ and decides whether 1s or 0s are in the majority. A **tie bandwidth** parameter $b \geq 0$ controls how strict the definition of "tie" is: if the absolute difference between the count of 1s and 0s is at most $b$, the result is declared a tie rather than a majority.

**Examples** (sequence length $n = 4$):

| Sequence | Count of 1s | $b=0$ | $b=1$ | $b=2$ |
|---|---|---|---|---|
| `0011` | 2 | TIE | TIE | TIE |
| `1010` | 2 | TIE | TIE | TIE |
| `1110` | 3 | MAJ-1 | TIE | TIE |
| `0000` | 0 | MAJ-0 | MAJ-0 | TIE |
| `1111` | 4 | MAJ-1 | MAJ-1 | TIE |

The tie bandwidth $b$ is only meaningful when $b < n/2$. If $b \geq n/2$, every possible sequence is a tie.

### 3.2 Formal Definition (Canonical)

Let $x \in \{0,1\}^n$, $n$ even. Define:

$$s(x) = \sum_{i=0}^{n-1} x_i \qquad d(x) = |2s(x) - n|$$

The **canonical majority rule** with tie bandwidth $b \in \mathbb{N}$, $0 \leq b < n/2$, is:

$$\text{maj}_b(x) =
\begin{cases}
\text{TIE} & \text{if } d(x) \leq b \\
1 & \text{if } d(x) > b \text{ and } s(x) > n/2 \\
0 & \text{if } d(x) > b \text{ and } s(x) < n/2
\end{cases}$$

This two-step rule is the single authoritative definition: **first check for tie** using $d(x) \leq b$; **if not a tie**, the majority is the sign of $s(x) - n/2$. All implementations must use this rule to ensure fixture reproducibility.

> **Note on equivalent thresholds:** The conditions $s(x) > n/2$ and $s(x) < n/2$ (given $d(x) > b$) are equivalent to $s(x) > n/2 + b/2$ and $s(x) < n/2 - b/2$ respectively. However, the two-step canonical form above avoids edge cases arising from the fractional threshold $b/2$ when $b$ is odd, and is the normative definition.

**Valid parameter range:** $L \bmod 2 = 0$ (length must be even); $0 \leq b < n/2$.

### 3.3 A-Side Majority

Input: $x \in \{0,1\}^n$, $n$ even; bandwidth $b$; tie-breaking value $c_{\text{tie}} \in \{0, 1\}$ (strategy-specific, see Section 5).

$$c =
\begin{cases}
\text{maj}_b(x) & \text{if } \text{maj}_b(x) \neq \text{TIE} \\
c_{\text{tie}} & \text{if } \text{maj}_b(x) = \text{TIE}
\end{cases}$$

In hybrid strategies, a tie may instead trigger a fallback mechanism (pyramid), rather than emitting $c_{\text{tie}}$. This is always stated explicitly in the strategy definition.

### 3.4 B-Side Majority

In the pure majority scheme, B ignores the index $j$ entirely:

$$f_B(j) = 0 \quad \Rightarrow \quad \hat{x}_j = c$$

---

## 4. Pyramid Algorithm

The pyramid algorithm uses PR boxes to compress A's input sequence into a single communication bit, while allowing B to compute a corresponding flip value from his index alone.

### 4.1 Structural Requirements

The input length must satisfy $L = 2^k$ for some $k \geq 1$. The algorithm uses exactly $L - 1$ PR boxes arranged in $k$ layers.

### 4.2 A-Side Pyramid

#### Verbal Description

A's input $x$ of length $L = 2^k$ is processed in $k$ layers, each reducing the sequence length by half. At each step, adjacent pairs of values are combined using a PR box.

For a pair $(p_0, p_1)$ at any unit:
- PR box input: $a = p_0 \oplus p_1$
- PR box A-side output: $o_A$
- Unit output: $p_0 \oplus o_A$

After $k$ layers, one bit remains at the apex. This is the communication bit $c$.

#### Formal Definition

Initialise: $u^{(0)}_i = x_i$ for $i = 0, \ldots, L-1$.

For layer $\ell = 0, \ldots, k-1$, for each $i = 0, \ldots, 2^{k-\ell-1} - 1$:

$$a^{(\ell)}_i = u^{(\ell)}_{2i} \oplus u^{(\ell)}_{2i+1}$$

$$o^{(\ell)}_i = \text{PR}_A(\ell,\, i,\, a^{(\ell)}_i)$$

$$u^{(\ell+1)}_i = u^{(\ell)}_{2i} \oplus o^{(\ell)}_i$$

Communication bit: $c = u^{(k)}_0$

#### Pseudocode

```
function pyramid_A(x[0..L-1]):
    u = x
    for ℓ = 0 to k-1:
        u_next = array of length len(u)/2
        for i = 0 to len(u)/2 - 1:
            a = u[2i] XOR u[2i+1]
            o = PR_A(ℓ, i, input=a)        // shared instance PR(ℓ,i)
            u_next[i] = u[2i] XOR o
        u = u_next
    return u[0]
```

### 4.3 B-Side Pyramid

#### Verbal Intuition (One-Hot View)

B's index $j$ can be thought of as a one-hot vector of length $L$ — a 1 at position $j$, zeros elsewhere. This vector propagates up the pyramid: at each layer, exactly one unit receives a non-zero input (the unit whose sub-tree contains $j$). All other units see $(0,0)$ input and produce output 0. B XORs the outputs of all active units across all $k$ layers to form the flip function $f_B(j)$.

This one-hot view is provided as **intuition only**. The canonical implementation is the binary-bit formulation below.

#### Canonical Formulation (Binary-Bit — Normative)

**The binary-bit formulation is the authoritative definition.** All implementations must use it. The one-hot description above is equivalent but may lead to implementation divergence (left/right convention mismatches, layer mapping errors) and should not be used as a coding reference.

Represent $j$ in binary: $j = \sum_{\ell=0}^{k-1} j_\ell \cdot 2^\ell$, where $j_0$ is the least significant bit.

At layer $\ell$:
- Active unit index: $i_\ell = \lfloor j / 2^{\ell+1} \rfloor$
- B-side PR box input: $b^{(\ell)} = j_\ell$ (the $\ell$-th bit of $j$)
- B-side PR box output: $o^{(\ell)}_B = \text{PR}_B(\ell,\, i_\ell,\, b^{(\ell)})$

All non-active units at layer $\ell$ receive input $b = 0$ and produce output 0.

Flip function:

$$f_B(j) = \bigoplus_{\ell=0}^{k-1} o^{(\ell)}_B$$

#### Pseudocode

```
function pyramid_B(j, k):
    f = 0
    for ℓ = 0 to k-1:
        b = bit(j, ℓ)                // ℓ-th bit of j in binary (j_ℓ)
        i = floor(j / 2^(ℓ+1))      // active unit index
        o = PR_B(ℓ, i, input=b)     // shared instance PR(ℓ,i), same as A used
        f = f XOR o
    return f
```

---

## 5. Hybrid Algorithms

Hybrid algorithms combine majority and pyramid. They exploit two complementary strengths: majority is effective on skewed inputs (many 1s or many 0s) but fails on balanced inputs; the pyramid handles all inputs uniformly but requires $L-1$ PR boxes and degrades with PR-box quality.

**Tie handling is strategy-specific** — each strategy below explicitly defines what A does on a tie and what consequences this has for B.

---

### 5.1 Pure Majority (Baseline)

**Parameter constraints:** $L \bmod 2 = 0$

**PR boxes used:** $0$

**Tie policy:** A sends $c_{\text{tie}}$ (fixed, typically 0). B is unaware of ties.

**A-side:** $c = \text{maj}_b(x)$; on tie $c = c_{\text{tie}}$.

**B-side:** $f_B(j) = 0$; guess $\hat{x}_j = c$.

*(Border case: setting $s = 0$ in either vertical hybrid below reduces to pure majority.)*

---

### 5.2 Pure Pyramid (Baseline)

**Parameter constraints:** $L = 2^k$

**PR boxes used:** $L - 1$

**Tie policy:** N/A (no majority step).

**A-side:** $c = \text{pyramid}_A(x)$.

**B-side:** $f_B(j) = \text{pyramid}_B(j, k)$.

**Success probability:**

$$P_{\text{success}} = P_{\text{pyr}}(k,\, p_{\text{rule}}) = \frac{1 + (2p_{\text{rule}} - 1)^k}{2}$$

*(Border case: setting $s = \log_2 L$ in either vertical hybrid below reduces to pure pyramid.)*

---

### 5.3 Horizontal Hybrid

**Parameter constraints:** $L_1 \bmod 2 = 0$; $L_2 = 2^k$; $L = L_1 + L_2$

**PR boxes used:** $L_2 - 1$

**Tie policy (strategy-specific):** A tie in the prefix **triggers the pyramid** on the postfix. B does not know which branch A took.

#### A-Side

```
if maj_b(x[0..L1-1]) != TIE:
    c = maj_b(prefix)
else:
    c = pyramid_A(x[L1..L-1])
```

#### B-Side

```
if j < L1:
    f_B = 0                            // prefix: pass communication bit through
else:
    f_B = pyramid_B(j - L1, k)        // postfix: apply pyramid flip
```

#### Edge Case: Tie with Prefix Query

When the prefix ties, $c$ encodes postfix information (the pyramid output). If B's index $j$ is in the prefix, B still uses $f_B = 0$ — B cannot know that A switched to pyramid. B's guess $\hat{x}_j = c$ is then uncorrelated with $x_j$. This is an **accepted loss** with success probability $\tfrac{1}{2}$.

Symmetrically, when no tie occurs, $c$ encodes prefix majority information. If B's index is in the postfix, B's pyramid flip is mismatched to the communication bit; again, success probability is $\tfrac{1}{2}$.

**Reverse horizontal hybrid:** Pyramid on prefix (length $2^k$), majority on postfix (length $L_1$, even). Same structure with roles swapped.

#### Success Probability

Let $P_{\text{tie}}$ = probability the prefix ties.

For $j$ in the prefix:

$$P_{\text{success}}(j \in \text{prefix}) = (1 - P_{\text{tie}}) \cdot P(\text{maj correct}) + P_{\text{tie}} \cdot \tfrac{1}{2}$$

For $j$ in the postfix:

$$P_{\text{success}}(j \in \text{postfix}) = (1 - P_{\text{tie}}) \cdot \tfrac{1}{2} + P_{\text{tie}} \cdot P_{\text{pyr}}(k,\, p_{\text{rule}})$$

---

### 5.4 Vertical Hybrid: Pyramid then Majority (pyr/maj)

A applies $s$ pyramid layers, then applies majority to the apex sequence. B applies $s$ pyramid layers to decode the flip.

**Parameter constraints:**
- $L = 2^k$
- $L / 2^s \bmod 2 = 0$ (apex sequence must be even-length for majority)
- $0 \leq s \leq k - 1$

**PR boxes used:** $L - L/2^s$ (first $s$ layers of the full pyramid)

**Tie policy (strategy-specific):** Tie in the apex majority causes A to send $c_{\text{tie}}$ (fixed). B is unaware of ties.

#### A-Side

Run pyramid for $s$ layers to obtain the intermediate sequence $u^{(s)}$ of length $L/2^s$:

$$c = \text{maj}_b(u^{(s)}) \quad \text{(on tie: } c = c_{\text{tie}}\text{)}$$

#### B-Side

$$f_B(j) = \bigoplus_{\ell=0}^{s-1} o^{(\ell)}_B$$

#### Pseudocode

```
function vertPyrMaj_A(x, s, b):
    u = x
    for ℓ = 0 to s-1:
        u = pyramid_layer_A(u, ℓ)
    return majority(u, b)          // tie → c_tie

function vertPyrMaj_B(j, s):
    f = 0
    for ℓ = 0 to s-1:
        b_bit = bit(j, ℓ)
        i = floor(j / 2^(ℓ+1))
        o = PR_B(ℓ, i, input=b_bit)
        f = f XOR o
    return f
```

#### Interpretation and Assumptions

Think of $u^{(s)}$ as $L/2^s$ "apex values," each summarising a block of $2^s$ input bits through $s$ pyramid layers. Majority then votes over these apex values.

> **Assumption (Apex Uniformity):** Due to uniform PR-box marginals, the apex bits $u^{(s)}_i$ are treated as i.i.d. uniform Bernoulli(1/2) variables, independently of the input $x$. This is an approximation — it holds exactly for ideal PR boxes and is used to justify applying the standard majority correctness formula (Section 6.3) to the apex sequence.

---

### 5.5 Vertical Hybrid: Majority then Pyramid (maj/pyr)

A partitions the input into $2^s$ equal blocks, applies majority to each, then applies a short pyramid to the majority outputs. B applies a block-level pyramid.

**Parameter constraints:**
- $L / 2^s \bmod 2 = 0$ (each block must be even-length for majority)
- $L \bmod 2^s = 0$ (equal-size blocks required)
- $0 \leq s \leq k$

**PR boxes used:** $2^s - 1$ (pyramid over the $2^s$ majority outputs)

**Tie policy (strategy-specific):** Tie in block $m$ causes $y_m = c_{\text{tie}}$ (fixed per block). B is unaware of ties.

#### A-Side

Partition $x$ into $2^s$ blocks of length $L/2^s$:

$$y_m = \text{maj}_b(x^{(m)}), \quad m = 0, \ldots, 2^s - 1 \quad \text{(on tie: } y_m = c_{\text{tie}}\text{)}$$

$$c = \text{pyramid}_A(y)$$

#### B-Side

Determine B's block index: $m = \lfloor j \cdot 2^s / L \rfloor$

Apply B-side pyramid to $m$ as a length-$2^s$ pyramid index:

$$f_B(j) = \text{pyramid}_B(m,\, s)$$

#### Pseudocode

```
function vertMajPyr_A(x, s, b, L):
    block_size = L / 2^s
    y = array of length 2^s
    for m = 0 to 2^s - 1:
        block = x[m*block_size .. (m+1)*block_size - 1]
        y[m] = majority(block, b)    // tie → c_tie
    return pyramid_A(y)

function vertMajPyr_B(j, s, L):
    block_size = L / 2^s
    m = floor(j / block_size)        // which block?
    return pyramid_B(m, s)           // pyramid over 2^s block indices
```

#### Interpretation

The majority pre-compresses each block into a single bit. The pyramid routes B to the correct block's majority result. Success requires both that majority correctly represents B's block and that the pyramid correctly routes B.

---

## 6. Success Probability Analysis

### 6.1 Correct Logical Value

Define the **correct logical value** $c^*$ as the value of $c$ that, combined with $f_B(j)$, would yield $\hat{x}_j = x_j$:

$$c^* = x_j \oplus f_B^*(j)$$

where $f_B^*(j)$ is the flip value B would produce under ideal conditions (perfect PR boxes, no tie). In other words:

$$c^* \text{ is correct} \iff c \oplus f_B(j) = x_j$$

Define event $C$: "$c = c^*$" (communication bit is correct for B's index).
Define event $P$: "$f_B(j) = f_B^*(j)$" (B's flip function is correct).

### 6.2 General Decomposition

$$P(\text{success}) = P(C \land P) + P(\neg C \land \neg P)$$

Under the independence assumption (see 6.4):

$$P(\text{success}) = P(C)\,P(P) + (1 - P(C))(1 - P(P))$$

Equivalently:

$$P_{\text{success}} = \frac{1}{2} + \left(P(C) - \frac{1}{2}\right)\left(2P(P) - 1\right)$$

This shows the **double-error compensation** property: errors in $c$ and $f_B$ cancel under XOR. The gain above $1/2$ is the product of each sub-scheme's individual advantage.

### 6.3 Pyramid Correctness

For a pyramid (or partial pyramid) of depth $s$ with quality $p_{\text{rule}}$:

$$P_{\text{pyr}}(s,\, p_{\text{rule}}) = \frac{1 + (2p_{\text{rule}} - 1)^s}{2}$$

**Derivation:** $f_B(j)$ is the XOR of $s$ independent PR-box outputs. Each output is individually correct with probability $p_{\text{rule}}$. The XOR of $s$ independent Bernoulli variables is correct iff an even number are wrong; by induction this gives the formula above.

### 6.4 Majority Correctness

Assume $x \sim \text{Uniform}\{0,1\}^n$. For a fixed index $j$:

$$P(\text{maj correct}) = P(\text{no tie}) \cdot P(\text{maj} = x_j \mid \text{no tie}) + P(\text{tie}) \cdot P(c_{\text{tie}} = x_j)$$

For bandwidth $b = 0$:

$$P(\text{no tie, maj} = x_j) = \sum_{t > n/2} \binom{n}{t} \frac{t}{n} \cdot \frac{1}{2^{n-1}} + \sum_{t < n/2} \binom{n}{t} \frac{n-t}{n} \cdot \frac{1}{2^{n-1}}$$

This expression sums over Hamming weights $t$, using exchangeability: $P(x_j = 1 \mid s(x) = t) = t/n$ (each position equally likely to hold any value). For general $b$, the tie region $d(x) \leq b$ must be excluded from the sum and handled separately via the tie-breaking rule.

### 6.5 Combined Success and Independence Assumption

For vertical hybrids, let $P_C$ = majority correctness and $P_P = P_{\text{pyr}}(s, p_{\text{rule}})$:

$$\boxed{P_{\text{success}} = P_C \cdot P_P + (1 - P_C)(1 - P_P)}$$

> **Independence caveat:** This decomposition assumes that events $C$ and $P$ are approximately independent. For the **pyr/maj hybrid**, this holds under the Apex Uniformity assumption (Section 5.4): apex bits are treated as i.i.d. uniform, making majority correctness independent of the PR-box randomness driving $P$. For the **maj/pyr hybrid**, the majority output bits $y_m$ are not uniformly distributed and depend on $x$, so $C$ and $P$ are not strictly independent. The decomposition remains a good approximation but **exact results should be validated empirically** for the maj/pyr case.

---

## 7. Boundary Cases and Parameter Summary

| $s$ | Vertical pyr/maj | Vertical maj/pyr |
|---|---|---|
| $s = 0$ | Pure majority | Pure majority |
| $s = \log_2 L$ | Pure pyramid | Pure pyramid |
| $0 < s < \log_2 L$ | Hybrid | Hybrid |

| $b$ | Effect |
|---|---|
| $b = 0$ | Strictest majority; tie only on perfect balance |
| $0 < b < n/2$ | Wider tie zone; more fallback activations |
| $b \geq n/2$ | Everything is a tie; behaviour dominated by $c_{\text{tie}}$ or fallback |

| $p_{\text{rule}}$ | Pyramid behaviour |
|---|---|
| $1$ | Perfect; pyramid always correct |
| $> 1/2$ | Beneficial; more layers → better (diminishing returns) |
| $1/2$ | Uncorrelated; pyramid gives random output |
| $< 1/2$ | Adversarial; more layers → worse |

---

## 8. Algorithm Constraints Summary

Implementations must enforce these constraints and reject invalid configurations:

| Algorithm | Formal constraints | PR boxes used |
|---|---|---|
| Pure majority | $L \bmod 2 = 0$ | $0$ |
| Pure pyramid | $L = 2^k$ | $L - 1$ |
| Horizontal hybrid | $L_1 \bmod 2 = 0$; $L_2 = 2^k$ | $L_2 - 1$ |
| Vertical pyr/maj | $L = 2^k$; $(L / 2^s) \bmod 2 = 0$; $0 \leq s < k$ | $L - L/2^s$ |
| Vertical maj/pyr | $L \bmod 2^s = 0$; $(L / 2^s) \bmod 2 = 0$ | $2^s - 1$ |

---

## 9. Validation and Testing

To validate an implementation against the analytical success probability:

1. Fix parameters $L$, $s$, $b$, $p_{\text{rule}}$, tie-breaking rule $c_{\text{tie}}$.
2. Verify all formal constraints in Section 8 are satisfied; reject invalid configurations.
3. Enumerate all $(x, j)$ pairs (or sample uniformly for large $L$).
4. For each $(x, j)$: simulate PR-box randomness using the shared-instance model (Section 2.2), run A-side and B-side algorithms, record whether $\hat{x}_j = x_j$.
5. Compare empirical success rate to the analytical prediction from Section 6.

Key numerical checks:
- $p_{\text{rule}} = 1$, $s = k$: $P_{\text{success}} = 1.0$ (pure pyramid, perfect boxes)
- $p_{\text{rule}} = 1/2$: $P_{\text{success}}$ = pure majority rate (pyramid contributes nothing)
- $s = 0$: $P_{\text{success}}$ matches pure majority formula
- maj/pyr hybrid: compare empirical result to Section 6.5 formula and verify any discrepancy is consistent with the independence approximation

---

## 10. Summary of Design Space

The hybrid family interpolates between two extremes:

- **Pure majority** uses no PR boxes but fails on balanced inputs
- **Pure pyramid** uses $L-1$ PR boxes and degrades with PR-box quality

Hybrid strategies trade off:
- PR boxes required ($0$ to $L-1$)
- Sensitivity to PR-box quality $p_{\text{rule}}$
- Robustness on balanced vs. skewed inputs
- Horizontal (input-adaptive switching) vs. vertical (structural composition)

All schemes share double-error compensation through the XOR structure. The horizontal hybrid switches strategies based on observed input; the vertical hybrids compose strategies structurally. The success formula $P_{\text{success}} = \frac{1}{2} + (P_C - \frac{1}{2})(2P_P - 1)$ makes clear that both sub-schemes must individually beat chance for the hybrid to improve over either baseline.
