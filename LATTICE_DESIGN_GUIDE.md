# Comprehensive Lattice Design Guidelines

## 0. Theoretical Foundation

### 0.1 Generalization-Preserving Regularization
The core theory constrains model complexity to ensure generalization:

**γ = (p × C) / N < 1**

Where:
- p = number of parameters per cell (e.g., number of features for beta)
- C = number of cells in the lattice
- N = training sample size

When γ < 1, the model has fewer effective parameters than data points, enabling generalization. When γ > 1, the model can memorize training data.

### 0.2 Computing Prior Scales
Use `generalization_preserving_scales()` to compute theory-consistent regularization:

```python
prior_scales = decomp.generalization_preserving_scales(
    noise_scale=1.0,    # Expected noise level
    total_n=N_train,    # Training set size
    c=0.5,              # Conservatism parameter (0.5 = moderate)
    per_component=True  # Separate scale per decomposition component
)
```

The method returns a dictionary of scales for each component (main effects, 2-way, 3-way, etc.). Higher-order components get smaller scales (more regularization).

### 0.3 Decomposition and Regularization
The additive decomposition θ = θ_global + θ_main + θ_2way + θ_3way + ... allows:
- Lower-order terms to capture broad patterns (less regularized)
- Higher-order terms to capture fine details (more regularized)
- Automatic complexity control without full tensor

### 0.4 Order Selection Theory
- **Order k** includes all interactions up to k-way
- More order = more flexibility but more parameters
- Theory adjusts regularization per order, so higher order is safe if cells are populated
- Rule of thumb: use order that keeps γ < 1 for each component

### 0.5 Applying Theory in Practice
1. **Before designing lattice**: Estimate target cell count as N / 50 for ~50 samples/cell
2. **Check γ for intercept**: C_intercept × 1 / N (should be << 1)
3. **Check γ for beta**: C_beta × p_features / N (should be < 1)
4. **Use theory-based priors**: Always call `generalization_preserving_scales()`
5. **Scale multiplier**: Use scale_multiplier=50 for relaxed regularization when γ << 1

### 0.6 Ordinal Smoothness Regularization
When bins come from continuous variables (age, income, etc.), they have natural ordering. Adjacent bins should have similar parameters.

**Why it matters:**
- Age bin 35-45 should be similar to bins 25-35 and 45-55
- This is NOT true for unordered categoricals (occupation A vs B have no adjacency)

**Smoothness approaches:**
1. **Difference penalty**: Penalize (θ_k - θ_{k-1})² for adjacent bins
2. **Random walk prior**: θ_k ~ N(θ_{k-1}, σ²) - each bin centered on previous
3. **Second-order smoothness**: Penalize (θ_{k-1} - 2θ_k + θ_{k+1})² for curvature

**Implementation:**
```python
def ordinal_smoothness_penalty(params, axis=0):
    """Penalize differences between adjacent bins along one axis."""
    diffs = jnp.diff(params, axis=axis)
    return jnp.sum(diffs ** 2)

def multi_axis_smoothness_penalty(params, n_ordinal_axes):
    """Apply smoothness along each ordinal axis in interaction terms.
    
    For a 2-way interaction with shape (bins_a, bins_b, features),
    penalizes along axis 0 (a) and axis 1 (b).
    """
    penalty = 0.0
    for axis in range(n_ordinal_axes):
        if params.shape[axis] > 1:
            penalty += ordinal_smoothness_penalty(params, axis=axis)
    return penalty
```

**When to apply:**
- **Main effects**: Single ordinal dimension → smooth along that axis
- **Interactions**: Multiple ordinal dimensions → smooth along EACH ordinal axis
- Binned continuous variables: age, income, hours, price, latents
- Time dimensions: hour, day, month (cyclical - consider wrap-around)
- NOT for unordered categoricals: occupation, workclass, native-country

**Example for 2-way interaction age(6) × hours(4):**
- Shape: (6, 4, n_features)
- Apply smoothness along axis 0 (age bins: penalize θ[i+1,j] - θ[i,j])
- Apply smoothness along axis 1 (hour bins: penalize θ[i,j+1] - θ[i,j])

---

## 1. Overall Architecture

### 1.1 Two-Lattice Structure
- **Intercept lattice**: Captures baseline heterogeneity across subgroups (who has higher/lower baseline probability)
- **Beta lattice**: Captures how coefficients vary (whose features matter more/less)
- Intercept lattice is typically richer (more dimensions) than beta lattice

### 1.2 Cell Count Targets
- Target 30-60 samples/cell average for intercept lattice
- Can go sparser (10-30) if using lower order decomposition
- Below 10 samples/cell: likely too sparse, regularization dominates

### 1.3 Order Selection
- Start with order 3 for intercept (main + 2-way + 3-way interactions)
- Start with order 2 for beta (main + 2-way)
- Full order only if cells are well-populated

### 1.4 Theory Constraint
- γ = (parameters × cells) / N < 1 for generalization-preserving regularization
- Use `generalization_preserving_scales()` method to compute appropriate priors

---

## 2. Variable Selection for Lattice

### 2.1 Strong candidates for intercept lattice:
- Categorical variables with strong main effects on outcome
- Demographics that define subpopulations (age, marital status, education level)
- Variables where the baseline rate varies substantially across levels
- Binary indicators with large effect (e.g., capital_loss > 0 had 50% vs 23% rate)

### 2.2 Strong candidates for beta lattice:
- Variables that modify how other features affect outcome
- Time/context variables (hour of day, day of week) - coefficients may vary by context
- Interaction anchors (if A×B interaction matters, put A in beta lattice, B as feature)

### 2.3 Keep as linear features only (not in lattice):
- High-cardinality categoricals (>10-15 levels): occupation, native-country
- Continuous variables without clear non-linear effects
- Variables where adding to lattice increased cell count without improving AUC
- Rare categories that would create sparse cells

---

## 3. Binning Continuous Variables

### 3.1 When to bin:
- Variable has non-linear relationship with outcome
- Variable will be used in lattice
- Clear inflection points in rate vs. variable plot

### 3.2 How to bin:
- **Quantile-based**: Equal sample counts per bin (preferred for skewed distributions)
- **Rate-aware**: Place cuts at inflection points where outcome rate changes
- Check contingency table: rate should differ meaningfully across bins

### 3.3 Number of bins:
- Start with 4-6 bins
- More bins = finer granularity but more cells
- Fewer bins = coarser but better populated cells
- Sweet spot depends on N and other lattice dimensions

### 3.4 Special cases:
- **Zeros are meaningful**: Create explicit zero bin (capital-gain: 0, low, mid, high)
- **Sparse tails**: Combine extreme bins (age 65+ as single bin)

---

## 4. Handling Categorical Variables

### 4.1 Low cardinality (2-7 levels):
- Good candidate for lattice dimension
- Use as-is or group if some levels are rare

### 4.2 Medium cardinality (8-15 levels):
- Consider grouping into meaningful super-categories
- Or keep as one-hot features, not in lattice

### 4.3 High cardinality (>15 levels):
- Do NOT put raw variable in lattice (creates too many cells)
- **Option 1**: One-hot encode as features (simple, no lattice interaction)
- **Option 2**: Group into ~5-10 meaningful categories if domain knowledge allows
- **Option 3**: Use latent embeddings (see Section 4.5)

### 4.5 Using latents for high-cardinality categoricals:
Instead of excluding high-cardinality variables from the lattice, learn a low-dimensional embedding and treat the latent dimensions as continuous variables:

1. **Learn embedding**: Fit an embedding for the categorical (e.g., via matrix factorization, neural embedding, or correspondence analysis)
2. **Extract first 2-3 latent dimensions**: These are continuous values capturing the main structure
3. **Treat each latent as a continuous variable**:
   - Bin into 3-4 quantile-based bins for lattice
   - Compute within-bin normalized version for regression features
4. **Use binned latents as lattice dimensions**
5. **Use normalized latents as features** (enables piecewise linear effects)

Example:
- occupation (14 levels) → 2D continuous embedding
- Latent 1: bin into 4 levels → lattice dim, plus latent1_norm as feature
- Latent 2: bin into 4 levels → lattice dim, plus latent2_norm as feature
- Result: 16 cells, 2 continuous features, captures occupation structure

Benefits:
- High-cardinality variables can participate in lattice interactions
- Captures similarity structure (similar occupations cluster in latent space)
- Piecewise linear effects on latent dimensions
- Manageable cell count

### 4.4 Grouping rare categories:
- Group categories with <5% of data into "other"
- Example: race → white, asian, other
- Preserves signal while avoiding sparse cells

---

## 5. Feature Engineering

### 5.1 Within-bin normalization:
Apply ONLY when variable is BOTH:
- Binned for use in lattice, AND
- Used as a regressor (appears in X)

Transform: position within bin scaled to [0, 1]
```python
def within_bin_normalize(values, bin_edges):
    """Transform continuous values to 0-1 based on position within their bin."""
    values = np.asarray(values, dtype=float)
    full_edges = np.concatenate([[values.min()], bin_edges, [values.max()]])
    bins = np.digitize(values, bin_edges)
    
    lower = full_edges[bins]
    upper = full_edges[bins + 1]
    width = np.where(upper - lower == 0, 1.0, upper - lower)
    
    return np.clip((values - lower) / width, 0, 1)
```

### 5.2 Raw vs. normalized:
- Include BOTH raw and within-bin normalized for binned variables
- Raw captures magnitude, normalized captures within-bin position
- Together they enable piecewise linear functions

### 5.3 Don't omit features:
- Low-dimensional, high-N datasets benefit from all features
- Even if variable isn't in lattice, include as linear feature
- One-hot encode all categoricals not in lattice

### 5.4 Interaction features:
- If interaction is important and not captured by lattice, add explicit X1*X2 feature
- Example: price_product = nsw_price × vic_price

---

## 6. Using Logistic Regression for Guidance

### 6.1 Fit LR first:
- Fit a standard logistic regression with all features
- Examine coefficient magnitudes and significance
- Large coefficients → important variables, candidates for lattice

### 6.2 Check for non-linearity:
- Bin continuous variables and cross-tab with outcome
- If rate varies non-monotonically across bins, variable has non-linear effect
- Non-linear variables benefit most from lattice representation

### 6.3 Check for interactions:
- Add interaction terms to LR (X1*X2)
- If interaction coefficient is significant, consider putting both variables in same lattice
- Or put one in lattice, one as feature, to capture the interaction

### 6.4 Baseline comparison:
- LR AUC is your floor - lattice model should beat it
- If lattice model is worse than LR, structure is wrong (too sparse, wrong variables)

---

## 7. Diagnostic Checks

### 7.1 Before training:
- Print cell counts: total cells, samples/cell average
- Check γ < 1 for each lattice
- Verify bin distributions are reasonable (no empty or near-empty bins)

### 7.2 Contingency table analysis:
- Cross-tab each binned variable vs. outcome
- Confirm rate varies meaningfully across bins
- If rate is flat, binning isn't adding value

### 7.3 After training:
- Compare to simpler baseline (logistic regression)
- If lattice model is worse, likely too sparse or wrong structure
- Check which components have largest learned values

---

## 8. Iteration Strategy

### 8.1 Start simple:
- Begin with 2-3 most important variables in intercept lattice
- Single variable or 2 variables in beta lattice
- Verify it beats logistic regression baseline

### 8.2 Add complexity gradually:
- Add one dimension at a time
- Check if AUC improves
- If adding dimension hurts, either:
  - Use coarser bins for that variable
  - Keep it as linear feature instead
  - The interaction isn't helping

### 8.3 Use LR coefficients for guidance:
- Variables with large LR coefficients are important
- Start with those in the lattice
- Add others incrementally based on AUC improvement

---

## 9. Common Pitfalls

### 9.1 Too many cells:
- Symptom: AUC drops when adding dimensions
- Fix: Coarser bins, fewer dimensions, or keep variable as linear feature

### 9.2 Wrong variables in lattice:
- Symptom: Lattice model similar to or worse than logistic regression
- Fix: Check contingency tables, use LR coefficients to guide selection

### 9.3 Omitting important features:
- Symptom: Much worse than tree methods
- Fix: Include all features as linear terms, even if not in lattice

### 9.4 Too fine binning:
- Symptom: Many cells with <10 samples
- Fix: Use fewer, coarser bins based on rate inflection points

### 9.5 Ignoring special values:
- Symptom: Missing obvious signal (e.g., zeros in capital-gain)
- Fix: Create explicit bin for special values

---

## 10. Data-Driven Order Selection

### 10.1 Cell Occupancy Check
Before committing to an order, verify sufficient samples exist at that interaction level:

```python
from itertools import combinations

def compute_max_order(latent_bins, min_samples=20):
    """Determine max order based on actual cell occupancy."""
    N, k = latent_bins.shape
    
    for order in range(1, k + 1):
        min_count = N
        for dims in combinations(range(k), order):
            cell_counts = {}
            for i in range(N):
                cell = tuple(latent_bins[i, d] for d in dims)
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if cell_counts:
                min_count = min(min_count, min(cell_counts.values()))
        
        if min_count < min_samples:
            return order - 1
    return k
```

### 10.2 Typical Cell Counts
For N=3000 with 8 bins per dimension:
- Order 1 (main effects): ~375 samples/bin
- Order 2 (pairwise): ~47 samples/cell (8²=64)
- Order 3 (3-way): ~6 samples/cell (8³=512) - **usually too sparse**

**Rule:** Use min_samples=20 threshold. Order 3 rarely viable unless N is large or bins are coarse.

### 10.3 Dynamic Order by Fold
Different CV folds may have different optimal orders due to data distribution. Either:
1. Cap at order 2 for consistency (recommended)
2. Let each fold select its own order (more variance)

---

## 11. Tiered Feature Treatment

### 11.1 Three-Tier Architecture
Instead of binary important/weak split, use three tiers based on zero-order |β|:

| Tier | Selection | Treatment | Components |
|------|-----------|-----------|------------|
| Top | p95+ | Order 2 lattice | Full pairwise |
| Mid | p80-95 | Order 1 lattice | Main effects only |
| Weak | <p80 | Global | Single coefficient |

### 11.2 When to Use
- **High-dimensional data** (p >> 100): Tiering reduces parameters
- **Limited N**: Focus complexity budget on top features
- **Interpretability**: Clear hierarchy of feature importance

### 11.3 Implementation Pattern
```python
abs_beta = np.abs(lr.coef_[0])
top_threshold = np.percentile(abs_beta, 95)
mid_threshold = np.percentile(abs_beta, 80)

top_features = np.where(abs_beta >= top_threshold)[0]
mid_features = np.where((abs_beta >= mid_threshold) & (abs_beta < top_threshold))[0]
weak_features = np.where(abs_beta < mid_threshold)[0]
```

---

## 12. Proper Cross-Validation

### 12.1 Feature Selection Within Folds
**Never select features on full data before CV** - this is data leakage.

```python
# WRONG - data leakage
features = select_top_features(X, y)  # Uses test data!
for train, test in kfold.split(X, y):
    model.fit(X[train, features], y[train])

# CORRECT - selection within each fold
for train, test in kfold.split(X, y):
    features = select_top_features(X[train], y[train])
    model.fit(X[train, features], y[train])
```

### 12.2 Binning Within Folds
Quantile bin edges should also be computed on training data only:
```python
for train, test in kfold.split(X, y):
    bin_edges = np.percentile(X[train], [25, 50, 75])  # Train only
    train_bins = np.digitize(X[train], bin_edges)
    test_bins = np.digitize(X[test], bin_edges)  # Apply train edges
```

---

## 13. Quick Reference

| Aspect | Recommended |
|--------|-------------|
| Samples/cell | 30-60 (min 10) |
| Intercept order | 3 |
| Beta order | 2 |
| Bins per continuous var | 4-6 |
| Max lattice dimensions | 4-5 before sparsity issues |
| γ constraint | < 1 |
| Categorical cardinality for lattice | ≤ 7-10 levels |
