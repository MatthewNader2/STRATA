"""
src/macro_harvester.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Dynamic Macro Harvesting, Port Composition & Monadic Code Synthesis (Phase 4 / T4.3).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, PortSignature, AlgebraicSignature
    from .unification import UnificationGate, ExecutionContext, Substitution
    from .tokenizer import CellTokenizer
    from .planner import LatticePlanner
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, PortSignature, AlgebraicSignature
    from unification import UnificationGate, ExecutionContext, Substitution
    from tokenizer import CellTokenizer
    from planner import LatticePlanner

logger = get_logger("macro_harvester")


class MacroHarvester:
    """
    Synthesizes and registers MacroCell abstractions from verified cell subgraphs,
    providing monadic variable wire threading and composite port derivation.
    """

    @classmethod
    def derive_composite_interface(
        cls, cells: List[Cell]
    ) -> Tuple[Dict[str, PortSignature], Dict[str, PortSignature], Dict[str, Dict[str, str]]]:
        """
        Derives external composite inputs, outputs, and internal port renamings.

        Preserves intermediate optional/config ports and resolves variable collisions
        by namespacing conflicting external inputs to their originating sub-cell.
        """
        first_cell = cells[0]
        last_cell = cells[-1]

        composite_inputs: Dict[str, PortSignature] = {}
        port_mappings: Dict[str, Dict[str, str]] = {c.cell_id: {} for c in cells}

        # 1. First cell inputs are external inputs
        for p_name, p_sig in first_cell.inputs.items():
            composite_inputs[p_name] = p_sig
            port_mappings[first_cell.cell_id][p_name] = p_name

        preceding_cells: List[Cell] = [first_cell]

        # 2. Intermediate cells: inspect inputs
        for c in cells[1:]:
            for p_name, p_sig in c.inputs.items():
                is_config_or_optional = (
                    getattr(p_sig, "optional", False)
                    or getattr(p_sig, "is_config", False)
                    or getattr(p_sig, "default", None) is not None
                    or getattr(p_sig, "has_default", False)
                )

                is_satisfied_internally = False
                if not is_config_or_optional:
                    is_satisfied_internally = any(
                        out_sig.signature.matches(p_sig.signature)
                        for prev_c in preceding_cells
                        for out_sig in prev_c.outputs.values()
                    )

                # Unfulfilled data inputs or configuration ports become external inputs
                if not is_satisfied_internally or is_config_or_optional:
                    target_name = p_name
                    if target_name in composite_inputs:
                        existing_sig = composite_inputs[target_name]
                        # Disambiguate if port signatures differ
                        if not (
                            hasattr(existing_sig, "signature")
                            and hasattr(p_sig, "signature")
                            and existing_sig.signature.matches(p_sig.signature)
                        ):
                            target_name = f"{c.cell_id}_{p_name}"

                    composite_inputs[target_name] = p_sig
                    port_mappings[c.cell_id][p_name] = target_name

            preceding_cells.append(c)

        # 3. Last cell outputs form the primary composite outputs
        composite_outputs: Dict[str, PortSignature] = dict(last_cell.outputs)

        # Capture any unconsumed auxiliary intermediate outputs
        for i, c in enumerate(cells[:-1]):
            for out_name, out_sig in c.outputs.items():
                consumed = any(
                    in_sig.signature.matches(out_sig.signature)
                    for next_c in cells[i + 1:]
                    for in_sig in next_c.inputs.values()
                )
                if not consumed:
                    aux_name = out_name if out_name not in composite_outputs else f"{c.cell_id}_{out_name}"
                    composite_outputs[aux_name] = out_sig

        return composite_inputs, composite_outputs, port_mappings

    @classmethod
    def synthesize_code_template(
        cls,
        cells: List[Cell],
        port_mappings: Optional[Dict[str, Dict[str, str]]] = None
    ) -> str:
        """
        Synthesizes a threaded code template from sub-cells using monadic wire threading.

        Chains step results through sequentially generated intermediate variables
        (_wire_0, _wire_1, ...), preserving intermediate computation state.
        """
        port_mappings = port_mappings or {}
        code_cells = [c for c in cells if getattr(c, "code_template", None) and c.code_template.strip()]
        if not code_cells:
            return ""

        if len(code_cells) == 1:
            c = code_cells[0]
            template = c.code_template
            for old_p, new_p in port_mappings.get(c.cell_id, {}).items():
                if old_p != new_p:
                    template = template.replace(f"{{{old_p}}}", f"{{{new_p}}}")
            return template

        threaded_blocks: List[str] = []
        num_code_cells = len(code_cells)

        for idx, c in enumerate(code_cells):
            raw = c.code_template.strip()
            cell_mapping = port_mappings.get(c.cell_id, {})

            # Replace disambiguated ports
            for old_p, new_p in cell_mapping.items():
                if old_p != new_p:
                    raw = raw.replace(f"{{{old_p}}}", f"{{{new_p}}}")

            # Thread primary input: step 0 reads external port, subsequent steps read prior wire
            if idx > 0:
                in_wire = f"_wire_{idx - 1}"
                raw = raw.replace("{input_var}", in_wire)
                raw = raw.replace("{input}", in_wire)
                raw = raw.replace("{in_var}", in_wire)

                # Dynamic carrier port detection from cell schema
                in_carrier = getattr(c, "primary_input_name", None)
                if not in_carrier and getattr(c, "primary_input", None):
                    in_carrier = getattr(c.primary_input, "name", None)
                if not in_carrier and hasattr(c, "inputs") and c.inputs:
                    for p_name, p_sig in c.inputs.items():
                        is_cfg = getattr(p_sig, "optional", False) or getattr(p_sig, "is_config", False)
                        if not is_cfg:
                            in_carrier = p_name
                            break
                    if not in_carrier:
                        in_carrier = next(iter(c.inputs.keys()))
                if in_carrier:
                    raw = raw.replace(f"{{{in_carrier}}}", in_wire)

            # Thread primary output: intermediate steps write to _wire_{idx}, last step writes to output port
            if idx < num_code_cells - 1:
                out_wire = f"_wire_{idx}"
                raw = raw.replace("{output_var}", out_wire)
                raw = raw.replace("{output}", out_wire)
                raw = raw.replace("{out_var}", out_wire)

                # Dynamic carrier port detection from cell schema
                out_carrier = getattr(c, "primary_output_name", None)
                if not out_carrier and getattr(c, "primary_output", None):
                    out_carrier = getattr(c.primary_output, "name", None)
                if not out_carrier and hasattr(c, "outputs") and c.outputs:
                    out_carrier = next(iter(c.outputs.keys()))
                if out_carrier:
                    raw = raw.replace(f"{{{out_carrier}}}", out_wire)

            threaded_blocks.append(f"# Step {idx + 1}: {c.cell_id}\n{raw}")

        return "\n\n".join(threaded_blocks)

    @classmethod
    def harvest_macro(
        cls,
        cell_ids: List[str],
        orchestrator: LatticeOrchestrator,
        macro_id: Optional[str] = None,
        domain_name: Optional[str] = None,
        docstring: Optional[str] = None
    ) -> MacroCell:
        """
        Harvests a composite MacroCell from a list of cell IDs.
        Verifies internal type-monadic consistency, derives outer port interfaces,
        synthesizes monadic execution wires, and registers the new MacroCell in the orchestrator.
        """
        if not cell_ids:
            raise ValueError("[MACRO_HARVESTER] Cannot create macro from empty cell list.")

        loaded = orchestrator.loaded_cells
        cells: List[Cell] = []
        for cid in cell_ids:
            c = loaded.get(cid)
            if not c:
                raise ValueError(f"[MACRO_HARVESTER] Cell '{cid}' not found in orchestrator.")
            cells.append(c)

        # 1. Verify internal compositional validity
        planner = LatticePlanner(orchestrator=orchestrator)
        chain: List[Cell] = [cells[0]]
        accum_sigma = Substitution()
        for c in cells[1:]:
            step_sigma = planner._verify_transition(chain, c, accum_sigma)
            if step_sigma is None:
                raise ValueError(
                    f"[MACRO_HARVESTER] Cell '{chain[-1].cell_id}' does not form a type-valid composition with '{c.cell_id}'."
                )
            accum_sigma = step_sigma
            chain.append(c)

        # 2. Derive Outer Interface & Port Mappings
        composite_inputs, composite_outputs, port_mappings = cls.derive_composite_interface(cells)

        first_cell = cells[0]
        last_cell = cells[-1]

        # Stage & Role resolution
        first_role = str(getattr(first_cell, "node_role", "") or getattr(first_cell, "role", "") or "").lower()
        last_role = str(getattr(last_cell, "node_role", "") or getattr(last_cell, "role", "") or "").lower()
        first_is_source = (first_role in ("source", "ingress", "initial") or len(getattr(first_cell, "inputs", {})) == 0)
        last_is_sink = (last_role in ("sink", "terminal", "egress", "consumer") or not getattr(last_cell, "outputs", None))

        if first_is_source and last_is_sink:
            resolved_stage = 2
        elif first_is_source:
            resolved_stage = 1
        elif last_is_sink:
            resolved_stage = 3
        else:
            resolved_stage = 2

        # 3. Synthesize Code Template with Monadic Wires
        synthesized_code = cls.synthesize_code_template(cells, port_mappings)

        # Aggregate tokens and keywords
        combined_keywords: List[str] = []
        combined_deps: List[str] = []
        for c in cells:
            for kw in getattr(c, "keywords", []):
                if kw not in combined_keywords:
                    combined_keywords.append(kw)
            for dep in getattr(c, "dependencies", []):
                if dep not in combined_deps:
                    combined_deps.append(dep)

        resolved_macro_id = macro_id or f"MACRO_{first_cell.cell_id}_{last_cell.cell_id}"
        resolved_domain = domain_name or first_cell.domain_name

        # Complete internal DAG mapping
        internal_topology = {cells[i].cell_id: [cells[i + 1].cell_id] for i in range(len(cells) - 1)}
        internal_topology[cells[-1].cell_id] = []

        macro_cell = MacroCell(
            cell_id=resolved_macro_id,
            stage=resolved_stage,
            keywords=combined_keywords,
            inputs=composite_inputs,
            outputs=composite_outputs,
            domain_name=resolved_domain,
            dependencies=combined_deps,
            code_template=synthesized_code,
            sub_cells=list(cell_ids),
            algorithmic_steps=[getattr(c, "docstring", c.cell_id) or c.cell_id for c in cells],
            internal_topology=internal_topology,
            docstring=docstring or f"Composite macro morphism: {' -> '.join(cell_ids)}",
            verified=True,
        )

        macro_cell._resolved_sub_cells = {c.cell_id: c for c in cells}

        # Register in orchestrator
        orchestrator.loaded_cells[macro_cell.cell_id] = macro_cell

        # Refresh index and adjacency maps
        try:
            orchestrator.build_topology()
        except Exception as exc:
            logger.warning(f"[MACRO_HARVESTER] Topology rebuild after harvest failed: {exc}")

        logger.info(f"[MACRO_HARVESTER] Successfully harvested and registered MacroCell '{resolved_macro_id}'")
        return macro_cell
