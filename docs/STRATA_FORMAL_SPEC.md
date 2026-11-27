# STRATA Formal Specification: Graded Substructural Type System & Symmetric Monoidal Category Semantics

## 1. Overview & Motivation

**STRATA** (*Symbolic Topological Routing Architecture for Typed Automata*) formulates domain-agnostic program synthesis as **proof search inside a Resource-Aware Symmetric Monoidal Category (SMC) over a 4-Component Graded/Indexed Port Type System**. Every external library, hardware SDK, or DSL is declared solely via hot-swappable JSON domain manifests conforming to canonical `TreeSchema` without hardcoding library names or domain logic in the synthesis kernel.

Every typed port in STRATA carries a 5-axis specification whose semantics collapse cleanly into a 4-component formal type object `CanonicalPortType` $\mathcal{T} = \langle \tau, \rho, (\sigma, \mu), \varphi \rangle$ plus an optional domain-qualification namespace:

$$\mathcal{T} = \langle \tau,\ \rho,\ (\sigma, \mu),\ \varphi \rangle$$

where:

| Component | Formal Structure | Manifest / Runtime Carrier |
| :--- | :--- | :--- |
| **$\tau$** | Parametric Carrier Type over Bounded Multi-Parent DAG Poset $(\mathcal{C}, \le_{\mathcal{C}}, \top, \bot)$ | `PortSchema.carrier`, `BoundedPoset` |
| **$\rho$** | Layout & Symbolic Shape Index `(rank, shape, dtype, channel_count, channel_axis, length, rank_polymorphic)` | `LayoutContract`, `unify_shapes`, `join_shapes` |
| **$(\sigma, \mu)$** | Substructural Typestate $\sigma$ over Multi-Parent DAG $(\mathcal{S}, \le_{\mathcal{S}}, \top_{\mathcal{S}})$ paired with Linear Multiplicity $\mu$ | `PortSchema.typestate` / `accepted_states`, `resource_mode` |
| **$\varphi$** | Decidable Numeric Interval & Predicate Refinement Lattice | `PortSchema.invariants`, `RefinementSet` |

---

## 2. The 4-Component Type Structure $\mathcal{T} = \langle \tau, \rho, (\sigma, \mu), \varphi \rangle$

### 2.1 Parametric Carrier Poset $(\mathcal{C}, \le_{\mathcal{C}}, \top, \bot)$

Domain manifests declare type subsumption edges in `TreeSchema.types` using single inheritance (`parent`) or multiple inheritance (`parents: [...]`). Their union forms a finite directed graph $G_{\mathcal{C}} = (V_{\mathcal{C}}, E_{\mathcal{C}})$ required to be acyclic (`BoundedPoset.validate_acyclic()`), completed by a top element $\top$ (`TOP_CARRIERS = {"", "any", "object", "*", "top", "unknown", "generic", "⊤"}`) and bottom element $\bot$ (`UNIVERSAL_BOTTOM_SYMBOLS = {"void", "unit", "never", "bottom", "⊥"}`).

Reflexive-transitive closure induces the partial order $A \le_{\mathcal{C}} B$ on ground carriers, extended structurally to parametric type constructors (`CarrierType`):

- **[Sub-Top / Bottom]**: $\forall \tau.\ \tau \le_{\mathcal{C}} \top$ and $\bot \le_{\mathcal{C}} \tau$.
- **[Sub-Var]**: Universal type variables $\alpha \in \{\text{single uppercase identifier}\}$ unify polymorphically during template/combinator signature checks under a substitution-consistent bound environment $\theta$ (`lo_bounds`/`hi_bounds`), rejecting inconsistent multi-occurrence instantiations (e.g., `Tuple[int, str] ≰ Tuple[T, T]`) and preventing collisions with domain carrier names.
- **[Sub-Optional]**: $\texttt{None} \le_{\mathcal{C}} \texttt{Optional}[T]$, $A \le_{\mathcal{C}} T \implies A \le_{\mathcal{C}} \texttt{Optional}[T]$, and $A \le_{\mathcal{C}} B \implies \texttt{Optional}[A] \le_{\mathcal{C}} \texttt{Optional}[B]$.
- **[Sub-Container (Covariant)]**: For covariant unary/binary/ternary constructors $F \in \{\texttt{list}, \texttt{tuple}, \texttt{set}, \texttt{sequence}, \texttt{iterable}, \texttt{iterator}, \texttt{file}, \texttt{contextmanager}, \texttt{type}, \texttt{collection}, \texttt{dict}, \texttt{mapping}\}$:
  $$F_1 \le_{\text{head}} F_2 \ \wedge\ \forall i.\ A_i \le_{\mathcal{C}} B_i \implies F_1[A_1,\dots,A_k] \le_{\mathcal{C}} F_2[B_1,\dots,B_k]$$
  with **safe forward erasure** $F_1[A_1,\dots,A_k] \le_{\mathcal{C}} F_2$ whenever $F_1 \le_{\text{head}} F_2$, and **restricted reverse erasure** $F_1 \le_{\mathcal{C}} F_2[B_1,\dots,B_k]$ *only* when every $B_i$ is $\top$, a type variable, or ellipsis `...` (so `list <= List[Any]`, whereas `list ≰ List[int]`). Special arity widening handles `File[M, F] <= File[F]` via trailing-format covariance.
- **[Sub-Tuple-Variadic]**: $(\forall i \le k.\ A_i \le_{\mathcal{C}} T) \implies \texttt{Tuple}[A_1,\dots,A_k] \le_{\mathcal{C}} \texttt{Tuple}[T, \dots]$.
- **[Sub-Callable (Contravariant/Covariant)]**:
  $$\frac{n = m \quad \forall i.\ B_i \le_{\mathcal{C}} A_i \quad R_1 \le_{\mathcal{C}} R_2}{\texttt{Callable}[[A_1,\dots,A_n], R_1] \le_{\mathcal{C}} \texttt{Callable}[[B_1,\dots,B_m], R_2]}$$
  with `Callable[..., R_1] <= Callable[[B_1..B_m], R_2]` whenever $R_1 \le_{\mathcal{C}} R_2$, and dual matching to `Callable[..., R_2]`.

Because $G_{\mathcal{C}}$ is a finite bounded DAG, minimal upper bounds and common lower bounds are computable by ancestor/descendant intersection (`BoundedPoset.join_carriers`, `BoundedPoset.meet_carriers`), backing control-flow branch type joins $\tau_1 \sqcup \tau_2$.

### 2.2 Layout & Symbolic Shape Index $\rho$

$$\rho = \langle r,\ \vec{d},\ \delta,\ c,\ a,\ \ell,\ p_{\text{poly}} \rangle$$

where $r \in \mathbb{N} \cup \{\bot\}$ (`rank`), $\vec{d} \in (\mathbb{N} \cup \text{Sym} \cup \{\bot\})^*$ (`shape`), $\delta$ (`dtype`), $c$ (`channel_count`), $a$ (`channel_axis`), $\ell \in \mathbb{N} \cup \text{Sym} \cup \{\bot\}$ (`length`), and $p_{\text{poly}} \in \mathbb{B}$ (`rank_polymorphic`).

Layout satisfaction $\rho_1 \sqsubseteq_{\rho} \rho_2$ adopts **gradual metadata matching** on omitted `None` axes (treating missing producer metadata as unconstrained while enforcing every declared constraint):
1. If $r_2 \in \mathbb{N}$ and $\neg p_{\text{poly},2}$, then $r_1 \in \mathbb{N} \implies r_1 = r_2$ (with concrete/wild dtype equality on $\delta$, channel count equality on $c$, channel axis equality on $a$).
2. Symbolic length compatibility: concrete integer equality $\ell_1, \ell_2 \in \mathbb{N} \implies \ell_1 = \ell_2$.
3. Shape vector unification `unify_shapes(d₁, d₂)` ($\neg p_{\text{poly},2}$): $|\vec{d}_1| = |\vec{d}_2|$, and there exists a single-valued environment substitution $\theta : \text{Sym}_2 \to \mathbb{N} \cup \text{Sym}_1$ (symmetric with $\text{Sym}_1$) such that $\forall i$:
   - Concrete equality: $d_{1,i}, d_{2,i} \in \mathbb{N} \implies d_{1,i} = d_{2,i}$.
   - Non-linear symbolic variable consistency: if a symbolic dimension $d_{2,i} \in \text{Sym}_2$ repeats at multiple axes (e.g., square constraint `["N", "N"]`), every corresponding concrete integer or symbolic dimension $d_{1,i}$ must bind to the exact same value or symbol (hence `[4, 4] ⊑ ["N", "N"]` and `["M", "M"] ⊑ ["N", "N"]`, whereas `[3, 4] ⋢ ["N", "N"]` and `["M", "K"] ⋢ ["N", "N"]`).

### 2.3 Domain-Namespaced Substructural Typestate $(\sigma, \mu)$

Typestates are canonically domain-qualified as `dom::state` (or universal top $\top_{\mathcal{S}} =$ `"any"`), organized in an acyclic multi-parent DAG $(\mathcal{S}, \le_{\mathcal{S}}, \top_{\mathcal{S}})$ declared across loaded domains via `typestates.states[].parent_state`. Each consumer port may also specify an explicit finite disjunctive acceptance set $A_{\text{acc}}$ (`accepted_states`). Compatibility holds iff:

$$\sigma_1 \le_{\mathcal{S}} \sigma_2 \ \vee\ \exists s \in A_{\text{acc}}.\ \sigma_1 \le_{\mathcal{S}} s$$

Each port carries a usage multiplicity (linear resource mode) $\mu$:
- `value`: unrestricted structural value (structural weakening & contraction admissible).
- `borrow`: shared non-consuming borrow; resource remains live in linear context $\Delta$.
- `acquire` ($+\text{Lin}$): allocates a fresh linear handle token into active context $\Delta$.
- `release` ($-\text{Lin}$): consumes a live linear handle from $\Delta$ into released set $\mathcal{R}$, forbidding subsequent use-after-release.

Canonical resource discipline restricts `acquire` ($+\text{Lin}$) strictly to linear handles with a matching destructor (`release`) in the library or its declared transitive imports; garbage-collected objects carry `value`, and borrowed child views bound to an owning live handle carry `borrow`.

### 2.4 Decidable Numeric Interval & Predicate Refinement Lattice $\varphi$

Every port invariant list compiles deterministically into a decidable refinement element `RefinementSet` $\varphi = \langle I, \text{nz}, \text{ne}, \delta_{\varphi}, P \rangle$:
- **Real/Integer Interval** $I = \langle \ell, s_\ell, u, s_u \rangle$ with lower bound $\ell \in \mathbb{R} \cup \{-\infty\}$ (`strict` vs `closed`), upper bound $u \in \mathbb{R} \cup \{+\infty\}$ (`strict` vs `closed`), parsed from canonical predicates (`min=`, `gte=`, `gt=`, `max=`, `lte=`, `lt=`).
- **Structural Boolean Flags**: `nonzero` ($\text{nz}$, implied automatically whenever $\ell > 0$, $\ell = 0 \wedge s_\ell$, $u < 0$, or $u = 0 \wedge s_u$) and `non_empty` ($\text{ne}$).
- **Refinement Dtype & Predicate Atom Set**: optional $\delta_\varphi$ and normalized side-predicate atoms $P$.

Refinement entailment $\varphi_1 \models \varphi_2$ is decidable in $O(|P_1| + |P_2|)$ time:
1. Lower tightness: $I_2.\ell > -\infty \implies (I_1.\ell > I_2.\ell) \vee (I_1.\ell = I_2.\ell \wedge (I_1.s_\ell \vee \neg I_2.s_\ell))$.
2. Upper tightness: $I_2.u < +\infty \implies (I_1.u < I_2.u) \vee (I_1.u = I_2.u \wedge (I_1.s_u \vee \neg I_2.s_u))$.
3. Nonzero & Non-empty entailment: $\varphi_2.\text{nz} \implies (\varphi_1.\text{nz} \vee I_1 \models x \neq 0)$ and $\varphi_2.\text{ne} \implies \varphi_1.\text{ne}$.
4. Dtype & Predicate subset: $\varphi_2.\delta_\varphi \implies (\varphi_1.\delta_\varphi = \varphi_2.\delta_\varphi \vee \rho_1.\delta = \varphi_2.\delta_\varphi)$ and $\varphi_2.P \subseteq \varphi_1.P$.

Conjoined literal values `param_val` bound at `port_binder` must satisfy $\varphi_{\text{port}}(\text{param\_val})$ (`RefinementSet.check_literal`), failing statically with `REFINEMENT_VIOLATION` when violated.

---

## 3. Symmetric Monoidal Category & Coercion Coherence ($\mathcal{C}_{\text{coerce}} \subseteq \mathcal{C}$)

### 3.1 Objects and Morphisms

Let $\mathcal{C}$ denote the Resource-Aware Symmetric Monoidal Category induced by all loaded STRATA domain trees:
- **Objects**: Finite tensor products of port types $X_1 \otimes \dots \otimes X_n$ with monoidal unit $I = \bot_{\text{void}}$. Structural subtyping and layout/refinement entailment induce canonical identity/subsumption arrows $X \to Y$ whenever `signatures_unify(X, Y)` holds.
- **Morphisms**: Every `CellSchema` $f$ denotes a resource-graded morphism:
  $$f : \bigotimes_{i=1}^{n} \mathcal{T}^{\text{in}}_i \multimap^{\varepsilon}_{m} \bigotimes_{j=1}^{k} \mathcal{T}^{\text{out}}_j$$
  parameterized by `mutation_type` $m \in \{\texttt{pure}, \texttt{immutable}, \texttt{io}, \texttt{in\_place}, \texttt{stateful}, \texttt{stateful\_allocation}, \texttt{combinator}, \texttt{sink}\}$, declared effects $\varepsilon$ (`effects`), optional polymorphic axis propagation (`polymorphic_propagation`), and optional bridge annotation (`bridge_profile`).

### 3.2 Strict Coercion Subcategory $\mathcal{C}_{\text{coerce}}$ & Two-Tier Coherence

A critical failure mode in naively typed synthesis graphs is treating *arbitrary unary semantic computations* (e.g., `NUMPY_MEAN_ALL`, `CRYPTO_SHA256_ONESHOT_HEX`, `CV2_ERODE`) as silent type-conversion bridges. STRATA strictly partitions morphisms via an explicit wide subcategory $\mathcal{C}_{\text{coerce}} \subseteq \mathcal{C}$ of **Representation Coercions** (`LiveBridgeGraph`):

$$f \in \text{Mor}(\mathcal{C}_{\text{coerce}}) \iff \begin{cases}
f.\texttt{bridge\_profile} \neq \bot \ \wedge\ f.\texttt{bridge\_profile.is\_bridge} = \top \\
m_f \in \{\texttt{pure}, \texttt{immutable}\} \\
\neg\text{has\_destructive\_effect}(f) \\
|\text{ReqData}(f)| = 1 \ \wedge\ \text{ReqAsset}(f) = \emptyset \ \wedge\ \text{ReqUnboundScalar}(f) = \emptyset \\
\mu(\text{in}) \in \{\texttt{value}, \texttt{borrow}\} \ \wedge\ \mu(\text{out}) \in \{\texttt{value}, \texttt{borrow}\} \\
\Delta(\text{in}, \text{out}) \neq \emptyset
\end{cases}$$

Within $\mathcal{C}_{\text{coerce}}$, edges partition cleanly into two **Coherence Tiers**:
- **Tier 1 (Lossless Canonical Isomorphisms / Embeddings)**: $\neg \texttt{lossy} \wedge \neg \texttt{requires\_intent}$ (e.g., `numpy::color_bgr` $\leftrightarrow$ `numpy::color_rgb` $\leftrightarrow$ `pillow::mode_rgb`, `numpy::gray` $\leftrightarrow$ `pillow::mode_grayscale`, `DataFrame` $\to$ `ndarray`). Automatically admissible whenever carrier/layout/typestate alignment requires zero information loss.
- **Tier 2 (Information-Losing or Intent-Bearing Projections)**: $\texttt{lossy} \vee \texttt{requires\_intent} \vee (r_{\text{out}} < r_{\text{in}})$ (e.g., `CV2_CVT_COLOR_BGR2GRAY`, `NUMPY_TOLIST`, `AUDIO_STEREO_TO_MONO`). Subject to **Coercion Coherence** (`LiveBridgeGraph.find_bridges`):
  1. Never constructed when `allow_lossy = False`.
  2. Permitted during planning *only* when the target consumer port explicitly constrains at least one changed axis in $\Delta(e)$ (e.g., consumer requires `typestate = "numpy::gray"` or explicit carrier conversion or concrete target rank/dtype/shape), preventing spurious rank-dropping or lossy silent downgrades on polymorphic $\top$ consumers.
  3. Penalized with strict lexicographic priority `(is_tier2, total_cost, hop_count)` (`is_tier2 = lossy ∨ requires_intent`) so Tier-1 lossless paths strictly dominate any Tier-2 path.

---

## 4. Typing Judgments & Substructural Inference Rules

Planning constructs well-typed derivations $\Gamma; \Delta; \mathcal{R} \vdash \vec{n} \dashv \Gamma'; \Delta'; \mathcal{R}'$ where $\Gamma : \text{Var} \to \mathcal{T}$ is the carrier/typestate/refinement environment, $\Delta \subseteq \text{Var}$ is the set of active linear handles (`acquire` minus `release`), and $\mathcal{R} \subseteq \text{Var}$ is the released-handle tombstones set.

### [T-Var / T-Sub] Subsumption & Alias Preservation
$$\frac{\Gamma(x) = \mathcal{T}_1 \quad x \notin \mathcal{R} \quad \text{signatures\_unify}(\mathcal{T}_1, \mathcal{T}_2)}{\Gamma; \Delta; \mathcal{R} \vdash x : \mathcal{T}_2 \dashv \Gamma; \Delta; \mathcal{R}}$$

### [T-Coerce] Multi-Hop Coercion Insertion
$$\frac{\Gamma(x) = \mathcal{T}_{\text{src}} \quad x \notin \mathcal{R} \quad p = [e_1; \dots; e_k] \in \mathcal{C}_{\text{coerce}}(\mathcal{T}_{\text{src}}, \mathcal{T}_{\text{dst}}) \quad \mathcal{T}' = \text{apply}^*_p(\mathcal{T}_{\text{src}})}{\Gamma; \Delta; \mathcal{R} \vdash p(x) : \mathcal{T}' \dashv \Gamma; \Delta; \mathcal{R}}$$
where $\text{apply}(e, \mathcal{T})$ respects `e.cell.polymorphic_propagation` (`preserves_layout`, `preserves_typestate`).

### [T-Call] Monoidal Multi-Output Application & Polymorphic Propagation
Let cell $f$ have required data inputs $p_1,\dots,p_n$ (excluding any port explicitly bound to a prompt-grounded literal/enum parameter in `IRNode.params`), optional data inputs optionally matched against remaining explicit `$var` inputs, non-data parameters grounded via prompt/defaults/slot-fill and checked against `RefinementSet.check_literal`, external assets strictly grounded from prompt (`UNBOUND_EXTERNAL_ASSET` / `MISSING_EXTERNAL_ASSET` otherwise), and output ports $q_1 : \mathcal{T}^{\text{out}}_1, \dots, q_m : \mathcal{T}^{\text{out}}_m$ bound to variables $y_1,\dots,y_m$:

$$\frac{\begin{array}{c}
\forall i.\ \Gamma; \Delta; \mathcal{R} \vdash \text{wire}(p_i) : \mathcal{T}^{\text{in}}_i \quad
\forall k.\ \varphi_{k}(\text{lit}_k) = \top \quad
\mathcal{T}'_j = \text{prop}_f(\mathcal{T}_{\text{src}}, \mathcal{T}^{\text{out}}_j) \\
\Delta_{\text{rel}} = \{v \mid (p_i \mapsto v) \wedge \mu(p_i) = \texttt{release}\} \quad
\Delta_{\text{acq}} = \{y_j \mid \mu(q_j) = \texttt{acquire}\}
\end{array}}{\Gamma; \Delta; \mathcal{R} \vdash (y_1,\dots,y_m) = f(\dots) \dashv \Gamma[y_j \mapsto \mathcal{T}'_j];\ (\Delta \setminus \Delta_{\text{rel}}) \cup \Delta_{\text{acq}};\ \mathcal{R} \cup \Delta_{\text{rel}}}$$

### [T-Mut] In-Place Typestate Update
When `f.mutation_type == "in_place"` and wired input variable $v$ unifies in carrier with produced output port $q_j : \mathcal{T}'_j$:
$$\Gamma[v \mapsto \mathcal{T}'_j]$$
updates the linear/stateful alias of $v$ directly in place (e.g., mutating `matplotlib::axes_initialized` to `matplotlib::axes_with_content`).

### [T-Branch] Balanced Linear Choice & Lattice Join
For structured `branch(cond, body, else_body)`:
- Both branches evaluate under active linear context $(\Gamma, \Delta, \mathcal{R})$ with `require_all_closed = False`.
- **Two-Branch Rule**: Requires $\Delta_{\text{then}} = \Delta_{\text{else}}$ (otherwise `UNBALANCED_BRANCH_RESOURCES`). Exit context becomes $\Delta' = \Delta_{\text{then}}$, $\mathcal{R}' = \mathcal{R}_{\text{then}} \cup \mathcal{R}_{\text{else}}$, and result type becomes the poset/interval join $\mathcal{T}_{\text{out}} = \mathcal{T}_{\text{then}} \sqcup \mathcal{T}_{\text{else}}$ (`join_port_types`).
- **One-Sided Branch (`else_body = []`)**: Requires $\Delta_{\text{then}} = \Delta \wedge \mathcal{R}_{\text{then}} = \mathcal{R}$ (otherwise `UNBALANCED_BRANCH_RESOURCES`), joining output type against pre-branch $\Gamma(\text{out})$ or widening carrier to `Optional[T]`.

### [T-Loop] Resource-Invariant Iteration
For structured `loop(cond, iter, body)`:
- Loop induction variable $x$ receives deduced element type `(rank = max(0, r - 1), carrier = dtype if r == 1 else container_carrier)`.
- Body evaluates under $(\Gamma[x \mapsto \mathcal{T}_x], \Delta, \mathcal{R})$ with `require_all_closed = False`.
- Requires **Resource Neutrality**: $\Delta_{\text{body}} = \Delta$ (no handle leaked across loop iterations) AND $(\mathcal{R}_{\text{body}} \setminus \mathcal{R}) \cap \Delta = \emptyset$ (no outer linear handle double-released inside a loop); otherwise fails with `UNBALANCED_LOOP_RESOURCES`.

---

## 5. Soundness Theorems & Proof Sketches

### Theorem 1 (Type, Shape & Refinement Preservation)
*If $\Gamma; \Delta; \mathcal{R} \vdash P \dashv \Gamma'; \Delta'; \mathcal{R}'$ succeeds cleanly in Layer 2/3, then every call's input carrier $\tau$, symbolic layout $\rho$ (rank, shape, dtype, length), typestate $\sigma$, and refinement invariants $\varphi$ are satisfied at binding time, and every bound constant literal satisfies $\varphi(\text{lit}) = \top$.*

**Proof Sketch**: By structural induction over the derivation tree using `[T-Call]`, `[T-Mut]`, `[T-Coerce]`, `[T-Branch]`, and `[T-Loop]`. `signatures_unify` verifies poset membership $A \le_{\mathcal{C}} B$ (multi-parent BFS + variance), shape substitution `unify_shapes`, typestate DAG inheritance, and `RefinementSet.entails`. Whenever `[T-Branch]` joins divergent paths via `join_port_types`, join computes the lattice LUB $(\tau_1 \sqcup \tau_2, \sigma_1 \sqcup \sigma_2, \varphi_1 \sqcup \varphi_2)$ and drops non-invariant shape axes, ensuring any downstream consumer accepting $\mathcal{T}_1 \sqcup \mathcal{T}_2$ accepts values produced by either runtime branch. Concrete literal bindings are checked eagerly in `bind_and_emit` via `RefinementSet.check_literal`. $\square$

### Theorem 2 (Substructural Linear Resource & Borrow-Lifetime Safety)
*For any top-level plan accepted with deliverable $d$, every handle acquired during execution is either returned as $d$ or released exactly once across all control-flow paths, no variable in $\mathcal{R}$ is dereferenced after release, and no borrowed child handle/view whose owning resource has been released is dereferenced after owner release.*

**Proof Sketch**: `[T-Var]` and `_check_var_live` enforce both $x \notin \mathcal{R}$ and $\forall o \in \text{owners}(x).\ o \notin \mathcal{R}$ prior to variable wiring and inside `loop`/`branch` headers and `condition` tokens. In `[T-Call]`, every port with $\mu = \texttt{release}$ moves its bound handle $v$ from $\Delta$ to $\mathcal{R}$ monotonically, while any `borrow` output or child handle derived from a live borrowed/acquired owner records its transitive `borrow_owners` tuple so owner release invalidates all dependent borrows. `[T-Branch]` enforces $\Delta_{\text{then}} = \Delta_{\text{else}}$, unions tombstones $\mathcal{R}_{\text{then}} \cup \mathcal{R}_{\text{else}}$ and borrow-owner dependencies, and forbids unsafe outer releases in single-arm branches. `[T-Loop]` enforces $\Delta_{\text{body}} = \Delta$ and forbids releasing outer handles in $\Delta$. At program exit (`require_all_closed = True`), top-level acceptance asserts $\Delta_{\text{exit}} \subseteq \{d\}$, guaranteeing zero leaked linear handles. $\square$

### Theorem 3 (Refinement & External Asset Honesty)
*No synthesis trace accepted by STRATA fabricates an external filesystem/network resource (`value_class = "external_asset"`) via LLM slot-filling or default guesswork, and no constant numeric literal violating a declared refinement bound passes Layer 3 emission.*

**Proof Sketch**: `IRParamValue._validate_asset_honesty` rejects `value_class == "external_asset" ∧ provenance == "estimated"`. Layer 2 `_check_external_assets` requires `provenance == "prompt"` for every required external asset port before planning succeeds. Layer 3 `make_slot_fill_fn` explicitly filters out `external_asset` ports, and `port_binder.bind_and_emit` refuses `UNBOUND_EXTERNAL_ASSET` without fallback and `REFINEMENT_VIOLATION` via `RefinementSet.check_literal`. $\square$

### Theorem 4 (Coercion Coherence)
*Every auto-inserted bridge edge $e$ belongs strictly to $\mathcal{C}_{\text{coerce}}$ (`bridge_profile.is_bridge = True`, pure/immutable, no destructive effects, linear-neutral), and Tier-2 lossy/intent-bearing coercions are never inserted without an explicit downstream axis constraint.*

**Proof Sketch**: `LiveBridgeGraph.build` filters strictly by `bp.is_bridge == True` and `mutation_type in ("pure", "immutable")` with zero destructive effects and `resource_mode in ("value", "borrow")`, ensuring domain computations never masquerade as silent type conversions. `LiveBridgeGraph.find_bridges` admits Tier-2 edges iff `allow_lossy` holds and at least one axis in `e.deltas` is explicitly constrained (strict non-subtype target carrier when `carrier` changes, non-`any` typestate, or non-null layout/invariant constraint) on `dst`. Edge cost ordering `(is_tier2, total_cost, hop_count)` ensures lossless Tier-1 paths are always preferred when both exist. $\square$

---

## 6. Specification-to-Implementation Mapping

| Formal Construct | Authoritative Implementation Module & Symbol |
| :--- | :--- |
| Type AST $\tau$ & Multi-Parent DAG Poset $(\mathcal{C}, \le_{\mathcal{C}}, \sqcup, \sqcap)$ | [src/type_system.py](../src/type_system.py): `CarrierType`, `parse_carrier`, `BoundedPoset` |
| Layout & Symbolic Shape Unification $\rho$ | [src/type_system.py](../src/type_system.py): `unify_shapes`, `join_shapes`; [src/bridge_graph.py](../src/bridge_graph.py): `CanonicalPortType` |
| Interval & Predicate Refinement Lattice $\varphi$ | [src/type_system.py](../src/type_system.py): `RefinementSet` |
| 4-Component Subsumption & Lattice Join | [src/bridge_graph.py](../src/bridge_graph.py): `signatures_unify`, `join_port_types`, `propagate_polymorphic_signature` |
| Coercion Subcategory $\mathcal{C}_{\text{coerce}}$ & Two-Tier Search | [src/bridge_graph.py](../src/bridge_graph.py): `LiveBridgeGraph.build`, `LiveBridgeGraph.find_bridges` |
| Tuple Destructuring Assignment Syntax | [src/frontends/formal_regex.py](../src/frontends/formal_regex.py): `parse_formal_prompt` |
| Substructural Linear Planner (`[T-Call]`, `[T-Mut]`, `[T-Branch]`, `[T-Loop]`) | [src/typed_planner.py](../src/typed_planner.py): `TypedIRPlanner._plan_block`, `TypedIRPlanner._wire_data_inputs` |
| Monoidal Emission & Literal Refinement Enforcement | [src/port_binder.py](../src/port_binder.py): `bind_and_emit` |
| Multi-Parent Poset Compilation & Loading | [src/lattice_db.py](../src/lattice_db.py): `load_compiled_lattice`; [src/strata_pipeline.py](../src/strata_pipeline.py): `StrataPipeline.from_trees` |
