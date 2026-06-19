#!/usr/bin/env python
# coding: utf-8
"""
Matched Molecular Pair Analysis (MMPA) toolkit powered by mmpdb.

Wraps the mmpdb CLI (fragment → index → loadprops → transform / predict / generate)
and exposes each step as an Agno tool for the Chemoinformatician agent.  Results are
stored in ``session_state["sar_analysis"]["mmps"]`` so the Report Generator agent
can consume them via ``prepare_mmpa_report_data``.
"""

import csv
import io
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from cs_copilot.storage import S3, OutputOperation, operation_rel_path
from cs_copilot.tools.io.session_memory import register_session_object

from .base_chemistry import BaseChemistryToolkit, ChemistryError

logger = logging.getLogger(__name__)

_MMPA_WORKSPACE_KEY = "mmpa_workspace"
_MMPA_DB_KEY = "mmpa_db_path"
_MMPA_RESULTS_KEY = "mmpa_results"
_MMPA_REPORT_DATA_KEY = "mmpa_report_data"


class MMPAError(ChemistryError):
    """Raised when an mmpdb operation fails."""


class MMPAToolkit(BaseChemistryToolkit):
    """Matched Molecular Pair Analysis toolkit wrapping the mmpdb CLI.

    Pipeline (each step is a separate tool):
    1. ``build_mmp_database``     – fragment + index + optionally load properties
    2. ``run_mmp_transform``      – enumerate analogues with Δprop statistics
    3. ``run_mmp_predict``        – predict Δprop between two specific molecules
    4. ``run_mmp_generate``       – generate novel structures via 1-cut rules
    5. ``prepare_mmpa_report_data`` – format all results for the Report Generator
    """

    def __init__(self):
        super().__init__("mmpa")
        self.register(self.build_mmp_database)
        self.register(self.run_mmp_transform)
        self.register(self.run_mmp_predict)
        self.register(self.run_mmp_generate)
        self.register(self.prepare_mmpa_report_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_mmpdb() -> None:
        result = subprocess.run(
            ["mmpdb", "--version"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise MMPAError(
                "mmpdb is not installed or not on PATH. "
                "Install it with: pip install mmpdb"
            )

    @staticmethod
    def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
        logger.debug("mmpdb cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise MMPAError(
                f"mmpdb failed (exit {result.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  stderr: {result.stderr.strip()}"
            )
        return result

    @staticmethod
    def _workspace(session_state: Optional[Dict[str, Any]]) -> Path:
        """Return the persistent local workspace for this session (creates on first call)."""
        if session_state is not None:
            existing = session_state.get(_MMPA_WORKSPACE_KEY)
            if existing and Path(existing).is_dir():
                return Path(existing)
        workspace = Path(tempfile.mkdtemp(prefix="cs_copilot_mmpa_"))
        if session_state is not None:
            session_state[_MMPA_WORKSPACE_KEY] = str(workspace)
        logger.info("MMPA workspace: %s", workspace)
        return workspace

    @staticmethod
    def _write_smiles(records: List[Dict[str, str]], path: Path) -> None:
        """Write [{smiles, id}] to a two-column tab-separated SMILES file."""
        with path.open("w") as fh:
            for r in records:
                fh.write(f"{r['smiles']}\t{r['id']}\n")

    @staticmethod
    def _write_props(records: List[Dict[str, Any]], path: Path) -> None:
        """Write [{id, prop1, prop2, ...}] to a tab-separated property file."""
        if not records:
            return
        fieldnames = list(records[0].keys())
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _parse_tsv(text: str) -> List[Dict[str, str]]:
        reader = csv.DictReader(io.StringIO(text.strip()), delimiter="\t")
        return list(reader)

    @staticmethod
    def _parse_list_stats(text: str) -> Dict[str, int]:
        """Extract compound/rule/pair counts from `mmpdb list` output."""
        stats: Dict[str, int] = {}
        for line in text.splitlines():
            parts = line.split()
            # expected: filename  #cmpds  #rules  #pairs  #envs  #stats  ...
            if len(parts) >= 4:
                try:
                    stats["num_compounds"] = int(parts[1])
                    stats["num_rules"] = int(parts[2])
                    stats["num_pairs"] = int(parts[3])
                except (ValueError, IndexError):
                    pass
                break
        return stats

    @staticmethod
    def _save_csv_artifact(
        rows: List[Dict[str, Any]],
        filename: str,
        session_state: Optional[Dict[str, Any]],
    ) -> str:
        """Write *rows* to a session-scoped CSV and return its storage path."""
        if not rows:
            return ""
        rel_path = operation_rel_path(
            OutputOperation.CHEMICAL_SPACE,
            "mmpa",
            filename,
            session_state=session_state,
            workflow_slug="mmpa",
        )
        fieldnames = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        with S3.open(rel_path, "w") as fh:
            fh.write(buf.getvalue())
        return S3.path(rel_path)

    @staticmethod
    def _detect_property_prefixes(headers: List[str]) -> List[str]:
        """Detect MMP property names from transform TSV column headers."""
        prefixes: List[str] = []
        for h in headers:
            if h.endswith("_from_smiles"):
                prefix = h[: -len("_from_smiles")]
                if prefix and prefix not in prefixes:
                    prefixes.append(prefix)
        return prefixes

    # ------------------------------------------------------------------
    # Public tools
    # ------------------------------------------------------------------

    def build_mmp_database(
        self,
        smiles_records: List[Dict[str, str]],
        property_records: Optional[List[Dict[str, Any]]] = None,
        db_name: str = "compounds",
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a Matched Molecular Pair database from a list of molecules.

        Runs three mmpdb steps in sequence:
        1. ``mmpdb fragment`` — cut each molecule into constant + variable parts.
        2. ``mmpdb index``    — find all pairs sharing the same constant scaffold.
        3. ``mmpdb loadprops``— (optional) attach measured activity/property values.

        The resulting ``.mmpdb`` SQLite file is stored in the session workspace and
        its path is saved to ``session_state["mmpa_db_path"]`` for downstream tools.

        Args:
            smiles_records: List of dicts with ``smiles`` and ``id`` keys,
                e.g. ``[{"smiles": "c1ccccc1O", "id": "phenol"}, ...]``.
            property_records: Optional list of dicts whose first key is ``id``
                followed by named property columns,
                e.g. ``[{"id": "phenol", "pIC50": 5.2, "logP": 1.5}, ...]``.
            db_name: Base filename for the database file (without extension).
            session_state: Agno-injected shared session state.

        Returns:
            Dict with ``db_path``, ``num_compounds``, ``num_rules``,
            ``num_pairs``, and ``properties`` (list of loaded property names).
        """
        self._require_mmpdb()
        if not smiles_records:
            raise MMPAError("smiles_records cannot be empty")

        ws = self._workspace(session_state)
        smi_path = ws / f"{db_name}.smi"
        fragdb_path = ws / f"{db_name}.fragdb"
        mmpdb_path = ws / f"{db_name}.mmpdb"
        props_path = ws / f"{db_name}_props.tsv"

        self._write_smiles(smiles_records, smi_path)

        logger.info("Fragmenting %d molecules…", len(smiles_records))
        self._run(["mmpdb", "fragment", str(smi_path), "-o", str(fragdb_path)])

        index_cmd = ["mmpdb", "index", str(fragdb_path), "-o", str(mmpdb_path)]
        if property_records:
            self._write_props(property_records, props_path)
            index_cmd += ["--properties", str(props_path)]

        logger.info("Indexing MMP database…")
        self._run(index_cmd)

        properties: List[str] = []
        if property_records:
            id_key = list(property_records[0].keys())[0]
            properties = [k for k in property_records[0] if k != id_key]

        list_result = self._run(["mmpdb", "list", str(mmpdb_path)], check=False)
        stats = self._parse_list_stats(list_result.stdout)

        result: Dict[str, Any] = {
            "db_path": str(mmpdb_path),
            "num_compounds": stats.get("num_compounds", len(smiles_records)),
            "num_rules": stats.get("num_rules", 0),
            "num_pairs": stats.get("num_pairs", 0),
            "properties": properties,
        }

        if session_state is not None:
            session_state[_MMPA_DB_KEY] = str(mmpdb_path)
            sar = session_state.get("sar_analysis") or {}
            sar.setdefault("mmps", {}).update({"db_info": result})
            session_state["sar_analysis"] = sar

        logger.info("MMP database built: %s", result)
        return result

    def run_mmp_transform(
        self,
        query_smiles: str,
        db_path: Optional[str] = None,
        property_names: Optional[List[str]] = None,
        min_pairs: int = 1,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enumerate analogues of a query molecule using MMP transformation rules.

        For each applicable rule in the database the method returns the resulting
        analogue SMILES together with predicted property changes derived from
        historical matched pairs (mean Δ, standard deviation, pair count, min, max).

        Args:
            query_smiles: SMILES of the molecule to transform.
            db_path: Path to a ``.mmpdb`` file.  If omitted, the path stored in
                ``session_state["mmpa_db_path"]`` is used.
            property_names: Properties to predict.  If omitted, all properties
                present in the database are evaluated.
            min_pairs: Minimum number of historical pairs to include a rule.
            session_state: Agno-injected shared session state.

        Returns:
            Dict with ``query_smiles``, ``transforms`` (list of dicts, each
            with ``analogue_smiles``, ``from_smiles``, ``to_smiles``, and
            ``{prop}_avg``, ``{prop}_std``, ``{prop}_count``, ``{prop}_min``,
            ``{prop}_max`` entries per property), and ``count``.
        """
        self._require_mmpdb()
        db_path = db_path or (session_state or {}).get(_MMPA_DB_KEY)
        if not db_path:
            raise MMPAError("db_path required — call build_mmp_database first.")

        cmd = [
            "mmpdb", "transform",
            "--smiles", query_smiles,
            "--min-pairs", str(min_pairs),
        ]
        if property_names:
            for prop in property_names:
                cmd += ["--property", prop]
        cmd.append(db_path)

        result = self._run(cmd)
        rows = self._parse_tsv(result.stdout)
        transforms = self._normalize_transform_rows(rows)

        csv_path = self._save_csv_artifact(
            [
                {
                    "query_smiles": query_smiles,
                    "analogue_smiles": t.get("analogue_smiles", ""),
                    "from_smiles": t.get("from_smiles", ""),
                    "to_smiles": t.get("to_smiles", ""),
                    **{k: v for k, v in t.items() if k not in ("analogue_smiles", "from_smiles", "to_smiles")},
                }
                for t in transforms
            ],
            "transform_results.csv",
            session_state,
        )

        output: Dict[str, Any] = {
            "query_smiles": query_smiles,
            "transforms": transforms,
            "count": len(transforms),
            "csv_path": csv_path,
        }

        if session_state is not None:
            sar = session_state.get("sar_analysis") or {}
            sar.setdefault("mmps", {})["transform_results"] = output
            session_state["sar_analysis"] = sar
            session_state[_MMPA_RESULTS_KEY] = output
            if transforms:
                register_session_object(
                    session_state,
                    "analysis",
                    {
                        "query_smiles": query_smiles,
                        "analysis_type": "mmp_transform",
                        "count": len(transforms),
                        "csv_path": csv_path,
                    },
                    label=f"MMP transform: {query_smiles[:40]}",
                    source_tool="mmpa.run_mmp_transform",
                )

        return output

    def run_mmp_predict(
        self,
        query_smiles: str,
        reference_smiles: str,
        property_name: str,
        db_path: Optional[str] = None,
        reference_value: Optional[float] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict the property change when converting a reference molecule to a query.

        Looks up the MMP transformation from ``reference_smiles`` to
        ``query_smiles`` in the database and returns the predicted Δprop (and
        absolute value when ``reference_value`` is supplied).

        Args:
            query_smiles: SMILES of the target (end-point) molecule.
            reference_smiles: SMILES of the starting molecule with a known property.
            property_name: Property to predict (must be loaded in the database).
            db_path: Path to the ``.mmpdb`` file.  Reads from session state if omitted.
            reference_value: Known property value of ``reference_smiles``; enables
                prediction of the absolute property value for the query.
            session_state: Agno-injected shared session state.

        Returns:
            Dict with ``query_smiles``, ``reference_smiles``, ``property_name``,
            ``predicted_delta``, ``std``, and optionally ``predicted_value``.
        """
        self._require_mmpdb()
        db_path = db_path or (session_state or {}).get(_MMPA_DB_KEY)
        if not db_path:
            raise MMPAError("db_path required — call build_mmp_database first.")

        cmd = [
            "mmpdb", "predict",
            "--smiles", query_smiles,
            "--reference", reference_smiles,
            "--property", property_name,
        ]
        if reference_value is not None:
            cmd += ["--value", str(reference_value)]
        cmd.append(db_path)

        result = self._run(cmd)
        output = self._parse_predict_output(
            result.stdout,
            query_smiles=query_smiles,
            reference_smiles=reference_smiles,
            property_name=property_name,
        )

        if session_state is not None:
            sar = session_state.get("sar_analysis") or {}
            predict_list = sar.setdefault("mmps", {}).setdefault("predict_results", [])
            predict_list.append(output)
            session_state["sar_analysis"] = sar

            csv_path = self._save_csv_artifact(
                predict_list,
                "predict_results.csv",
                session_state,
            )
            register_session_object(
                session_state,
                "analysis",
                {
                    "query_smiles": query_smiles,
                    "reference_smiles": reference_smiles,
                    "property_name": property_name,
                    "analysis_type": "mmp_predict",
                    "predicted_delta": output.get("predicted_delta"),
                    "predicted_value": output.get("predicted_value"),
                    "csv_path": csv_path,
                },
                label=f"MMP predict: {property_name} ({query_smiles[:30]})",
                source_tool="mmpa.run_mmp_predict",
            )

        return output

    def run_mmp_generate(
        self,
        query_smiles: str,
        db_path: Optional[str] = None,
        min_pairs: int = 1,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate novel molecules by applying 1-cut MMP rules to a query molecule.

        Each applicable transformation rule from the database is applied to the
        query structure to generate unique analogues.  The ``num_pairs`` field
        on each generated molecule indicates how many historical matched pairs
        support the rule.

        Args:
            query_smiles: SMILES of the starting molecule.
            db_path: Path to the ``.mmpdb`` file.  Reads from session state if omitted.
            min_pairs: Minimum number of historical pairs required to apply a rule.
            session_state: Agno-injected shared session state.

        Returns:
            Dict with ``query_smiles``, ``generated`` (list of dicts with
            ``smiles``, ``from_smiles``, ``to_smiles``, ``num_pairs``), and
            ``count``.
        """
        self._require_mmpdb()
        db_path = db_path or (session_state or {}).get(_MMPA_DB_KEY)
        if not db_path:
            raise MMPAError("db_path required — call build_mmp_database first.")

        cmd = [
            "mmpdb", "generate",
            "--smiles", query_smiles,
            "--min-pairs", str(min_pairs),
            "--columns", "start,from_smiles,to_smiles,final,#pairs",
            db_path,
        ]

        result = self._run(cmd)
        generated = self._parse_generate_output(result.stdout)

        csv_path = self._save_csv_artifact(
            [{"query_smiles": query_smiles, **g} for g in generated],
            "generate_results.csv",
            session_state,
        )

        output: Dict[str, Any] = {
            "query_smiles": query_smiles,
            "generated": generated,
            "count": len(generated),
            "csv_path": csv_path,
        }

        if session_state is not None:
            sar = session_state.get("sar_analysis") or {}
            sar.setdefault("mmps", {})["generate_results"] = output
            session_state["sar_analysis"] = sar
            session_state[_MMPA_RESULTS_KEY] = output
            if generated:
                register_session_object(
                    session_state,
                    "analysis",
                    {
                        "query_smiles": query_smiles,
                        "analysis_type": "mmp_generate",
                        "count": len(generated),
                        "csv_path": csv_path,
                    },
                    label=f"MMP generate: {query_smiles[:40]}",
                    source_tool="mmpa.run_mmp_generate",
                )

        return output

    def prepare_mmpa_report_data(
        self,
        title: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format all MMPA results in session state for the Report Generator agent.

        Reads results from ``session_state["sar_analysis"]["mmps"]`` (populated by
        the other MMPA tools) and assembles a ``save_rich_report``-compatible
        payload with sections, data tables, and molecule-structure figures for the
        query and top analogues.

        The returned dict is also written to ``session_state["mmpa_report_data"]``
        so the Report Generator agent can retrieve it directly.  Pass the dict to
        ``save_rich_report`` to produce HTML, PDF, or Markdown outputs.

        Args:
            title: Optional report title.  Defaults to
                ``"Matched Molecular Pair Analysis Report"``.
            session_state: Agno-injected shared session state.

        Returns:
            Dict with ``title``, ``summary``, ``sections``, ``figures``, and
            ``report_type`` suitable for ``save_rich_report``.
        """
        if session_state is None:
            raise MMPAError("session_state is required to read MMPA results.")

        sar = session_state.get("sar_analysis") or {}
        mmps = sar.get("mmps") or {}
        if not mmps:
            raise MMPAError(
                "No MMPA results found. Run run_mmp_transform or run_mmp_generate first."
            )

        report_title = title or "Matched Molecular Pair Analysis Report"
        summary = self._build_summary(mmps)
        sections = self._build_sections(mmps)
        figures = self._build_top_level_figures(mmps)

        payload: Dict[str, Any] = {
            "title": report_title,
            "summary": summary,
            "sections": sections,
            "figures": figures,
            "report_type": "mmpa",
        }

        session_state[_MMPA_REPORT_DATA_KEY] = payload
        return payload

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _normalize_transform_rows(self, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert raw TSV rows from ``mmpdb transform`` into structured records."""
        if not rows:
            return []
        headers = list(rows[0].keys())
        prop_prefixes = self._detect_property_prefixes(headers)

        transforms: List[Dict[str, Any]] = []
        for row in rows:
            smiles = row.get("SMILES") or row.get("smiles") or ""
            if not smiles:
                continue

            first_prop = prop_prefixes[0] if prop_prefixes else None
            record: Dict[str, Any] = {
                "analogue_smiles": smiles,
                "from_smiles": row.get(f"{first_prop}_from_smiles", "") if first_prop else "",
                "to_smiles": row.get(f"{first_prop}_to_smiles", "") if first_prop else "",
            }

            for prefix in prop_prefixes:
                for stat in ("avg", "std", "count", "min", "max"):
                    col = f"{prefix}_{stat}"
                    if col in row:
                        raw = row[col]
                        try:
                            record[col] = float(raw) if raw else None
                        except (ValueError, TypeError):
                            record[col] = None

            transforms.append(record)
        return transforms

    @staticmethod
    def _parse_predict_output(
        text: str,
        *,
        query_smiles: str,
        reference_smiles: str,
        property_name: str,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "query_smiles": query_smiles,
            "reference_smiles": reference_smiles,
            "property_name": property_name,
            "predicted_delta": None,
            "predicted_value": None,
            "std": None,
        }
        for line in text.splitlines():
            line = line.strip()
            if "predicted delta:" not in line:
                continue
            parts = line.split()
            try:
                delta_idx = parts.index("delta:") + 1
                output["predicted_delta"] = float(parts[delta_idx])
            except (ValueError, IndexError):
                pass
            if "predicted value:" in line:
                try:
                    val_idx = parts.index("value:", parts.index("delta:") + 2) + 1
                    output["predicted_value"] = float(parts[val_idx])
                except (ValueError, IndexError):
                    pass
            if "+/-" in line:
                try:
                    std_idx = parts.index("+/-") + 1
                    output["std"] = float(parts[std_idx])
                except (ValueError, IndexError):
                    pass
        return output

    @staticmethod
    def _parse_generate_output(text: str) -> List[Dict[str, Any]]:
        """Parse ``mmpdb generate --columns start,from_smiles,to_smiles,final,#pairs`` output."""
        generated: List[Dict[str, Any]] = []
        seen: set = set()
        # Split on the blank line that separates the two blocks; take the first block.
        first_block = text.split("\n\n")[0].strip()
        reader = csv.DictReader(io.StringIO(first_block), delimiter="\t")
        for row in reader:
            smiles = row.get("final") or row.get("SMILES") or ""
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            try:
                num_pairs = int(row.get("#pairs", 1))
            except (ValueError, TypeError):
                num_pairs = 1
            generated.append(
                {
                    "smiles": smiles,
                    "from_smiles": row.get("from_smiles", ""),
                    "to_smiles": row.get("to_smiles", ""),
                    "num_pairs": num_pairs,
                }
            )
        return generated

    # ------------------------------------------------------------------
    # Report-builder helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(mmps: Dict[str, Any]) -> List[str]:
        summary: List[str] = []

        db_info = mmps.get("db_info") or {}
        if db_info:
            summary.append(
                f"MMP database: {db_info.get('num_compounds', '?')} compounds, "
                f"{db_info.get('num_rules', '?')} rules, "
                f"{db_info.get('num_pairs', '?')} pairs."
            )

        transform = mmps.get("transform_results") or {}
        if transform:
            query = transform.get("query_smiles", "")
            count = transform.get("count", 0)
            summary.append(
                f"{count} analogue(s) identified for query {query} via MMP rules."
            )
            transforms = transform.get("transforms") or []
            prop_prefixes = sorted({
                k.replace("_avg", "")
                for t in transforms
                for k in t
                if k.endswith("_avg") and isinstance(t[k], float)
            })
            if prop_prefixes:
                summary.append(f"Properties predicted: {', '.join(prop_prefixes)}.")

        generate = mmps.get("generate_results") or {}
        if generate:
            summary.append(
                f"{generate.get('count', 0)} novel structure(s) generated "
                f"from {generate.get('query_smiles', '?')} by applying MMP rules."
            )

        predict = mmps.get("predict_results") or []
        if predict:
            summary.append(f"{len(predict)} pairwise property prediction(s) performed.")

        if not summary:
            summary.append("Matched Molecular Pair Analysis completed.")
        return summary

    @staticmethod
    def _mol_figure(smiles: str, name: str, caption: str) -> Dict[str, Any]:
        return {
            "structure_smiles": smiles,
            "structure_type": "molecule",
            "structure_name": name,
            "caption": caption,
        }

    def _build_sections(self, mmps: Dict[str, Any]) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []

        # ── Overview ──────────────────────────────────────────────────
        transform = mmps.get("transform_results") or {}
        generate = mmps.get("generate_results") or {}
        query_smiles = (
            transform.get("query_smiles") or generate.get("query_smiles") or ""
        )
        overview_paragraphs = [
            "Matched Molecular Pair (MMP) analysis systematically identifies pairs of "
            "molecules that differ by a single, well-defined structural modification. "
            "Transformation rules learned from the compound database are applied to "
            "the query structure to suggest analogues and predict property changes."
        ]
        if query_smiles:
            overview_paragraphs.append(f"Query structure submitted: {query_smiles}")

        sections.append(
            {
                "heading": "Overview",
                "paragraphs": overview_paragraphs,
                "figures": (
                    [
                        self._mol_figure(
                            query_smiles,
                            "Query molecule",
                            f"Query molecule for MMP analysis: {query_smiles}",
                        )
                    ]
                    if query_smiles
                    else []
                ),
                "tables": [],
            }
        )

        # ── Transform Results ─────────────────────────────────────────
        if transform:
            transforms = transform.get("transforms") or []
            prop_prefixes = sorted({
                k.replace("_avg", "")
                for t in transforms
                for k in t
                if k.endswith("_avg")
            })

            columns = ["Analogue SMILES", "Transformation"]
            for prop in prop_prefixes:
                columns += [f"{prop} Δavg", f"{prop} Δstd", f"{prop} N"]

            rows: List[Dict[str, str]] = []
            for t in transforms:
                row: Dict[str, str] = {
                    "Analogue SMILES": t.get("analogue_smiles", ""),
                    "Transformation": (
                        f"{t.get('from_smiles', '')} → {t.get('to_smiles', '')}"
                    ),
                }
                for prop in prop_prefixes:
                    avg = t.get(f"{prop}_avg")
                    std = t.get(f"{prop}_std")
                    cnt = t.get(f"{prop}_count")
                    row[f"{prop} Δavg"] = f"{avg:.3f}" if avg is not None else ""
                    row[f"{prop} Δstd"] = f"{std:.3f}" if std is not None else ""
                    row[f"{prop} N"] = str(int(cnt)) if cnt is not None else ""
                rows.append(row)

            figures = [
                self._mol_figure(
                    t["analogue_smiles"],
                    f"Analogue {idx}",
                    (
                        f"Analogue {idx}: {t['analogue_smiles']} "
                        f"(rule: {t.get('from_smiles', '')} → {t.get('to_smiles', '')})"
                    ),
                )
                for idx, t in enumerate(transforms[:10], start=1)
                if t.get("analogue_smiles")
            ]

            sections.append(
                {
                    "heading": "MMP Transform Results",
                    "paragraphs": [
                        f"{len(transforms)} analogue(s) were identified by applying "
                        "matched molecular pair rules to the query structure. "
                        "Δavg is the mean property change across historical pairs; "
                        "Δstd is the standard deviation; N is the number of supporting pairs."
                    ],
                    "figures": figures,
                    "tables": [
                        {
                            "title": "Transform Results",
                            "columns": columns,
                            "rows": rows,
                        }
                    ],
                }
            )

        # ── Generated Structures ──────────────────────────────────────
        if generate:
            generated = generate.get("generated") or []
            columns_gen = ["Generated SMILES", "Rule (from → to)", "Num Pairs"]
            rows_gen = [
                {
                    "Generated SMILES": g.get("smiles", ""),
                    "Rule (from → to)": (
                        f"{g.get('from_smiles', '')} → {g.get('to_smiles', '')}"
                    ),
                    "Num Pairs": str(g.get("num_pairs", "")),
                }
                for g in generated
            ]
            figures_gen = [
                self._mol_figure(
                    g["smiles"],
                    f"Generated structure {idx}",
                    (
                        f"Generated structure {idx}: {g['smiles']} "
                        f"(supported by {g.get('num_pairs', '?')} historical pairs)"
                    ),
                )
                for idx, g in enumerate(generated[:10], start=1)
                if g.get("smiles")
            ]
            sections.append(
                {
                    "heading": "Generated Structures",
                    "paragraphs": [
                        f"{len(generated)} novel structure(s) were generated by applying "
                        "1-cut MMP rules to the query molecule. "
                        "'Num Pairs' indicates the number of historical pairs supporting each rule."
                    ],
                    "figures": figures_gen,
                    "tables": [
                        {
                            "title": "Generated Structures",
                            "columns": columns_gen,
                            "rows": rows_gen,
                        }
                    ],
                }
            )

        # ── Property Predictions ──────────────────────────────────────
        predict_results = mmps.get("predict_results") or []
        if predict_results:
            columns_pred = [
                "Query SMILES", "Reference SMILES", "Property",
                "Δprop", "Std", "Predicted value",
            ]
            rows_pred = [
                {
                    "Query SMILES": p.get("query_smiles", ""),
                    "Reference SMILES": p.get("reference_smiles", ""),
                    "Property": p.get("property_name", ""),
                    "Δprop": (
                        f"{p['predicted_delta']:.3f}"
                        if p.get("predicted_delta") is not None
                        else ""
                    ),
                    "Std": f"{p['std']:.3f}" if p.get("std") is not None else "",
                    "Predicted value": (
                        f"{p['predicted_value']:.3f}"
                        if p.get("predicted_value") is not None
                        else ""
                    ),
                }
                for p in predict_results
            ]
            sections.append(
                {
                    "heading": "Property Predictions",
                    "paragraphs": [
                        "MMP-based property predictions for the specified query/reference pairs."
                    ],
                    "figures": [],
                    "tables": [
                        {
                            "title": "Property Predictions",
                            "columns": columns_pred,
                            "rows": rows_pred,
                        }
                    ],
                }
            )

        return sections

    @staticmethod
    def _build_top_level_figures(mmps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Top-level figures are empty — all molecule figures live inside sections."""
        return []


__all__ = ["MMPAToolkit", "MMPAError"]
