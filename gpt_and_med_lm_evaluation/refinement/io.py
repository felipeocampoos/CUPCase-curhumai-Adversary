"""
I/O utilities for JSONL logging and loading refinement traces.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Iterator, Any, Dict, Union
from datetime import datetime

from .schema import RefinementTrace


class JSONLLogger:
    """Logger for writing refinement traces to JSONL format."""
    
    def __init__(
        self,
        output_path: Union[str, Path],
        append: bool = False,
    ):
        """
        Initialize JSONL logger.
        
        Args:
            output_path: Path to output JSONL file
            append: If True, append to existing file; if False, overwrite
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._file = None
        self._count = 0
    
    def __enter__(self) -> "JSONLLogger":
        mode = "a" if self.append else "w"
        self._file = open(self.output_path, mode, encoding="utf-8")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            self._file = None
    
    def log(self, trace: RefinementTrace) -> None:
        """
        Log a single refinement trace.
        
        Args:
            trace: RefinementTrace to log
        """
        if self._file is None:
            raise RuntimeError("Logger not opened. Use 'with' statement.")
        
        data = trace.to_dict()
        data["_logged_at"] = datetime.now().isoformat()
        data["_index"] = self._count
        
        line = json.dumps(data, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        self._count += 1
    
    def log_dict(self, data: Dict[str, Any]) -> None:
        """
        Log arbitrary dictionary data.
        
        Args:
            data: Dictionary to log
        """
        if self._file is None:
            raise RuntimeError("Logger not opened. Use 'with' statement.")
        
        data = dict(data)
        data["_logged_at"] = datetime.now().isoformat()
        data["_index"] = self._count
        
        line = json.dumps(data, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        self._count += 1


def load_refinement_traces(
    input_path: Union[str, Path],
) -> List[RefinementTrace]:
    """
    Load refinement traces from JSONL file.
    
    Args:
        input_path: Path to JSONL file
        
    Returns:
        List of RefinementTrace objects
    """
    traces = []
    input_path = Path(input_path)
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            trace = RefinementTrace.from_dict(data)
            traces.append(trace)
    
    return traces


def iterate_refinement_traces(
    input_path: Union[str, Path],
) -> Iterator[RefinementTrace]:
    """
    Iterate over refinement traces from JSONL file without loading all into memory.
    
    Args:
        input_path: Path to JSONL file
        
    Yields:
        RefinementTrace objects one at a time
    """
    input_path = Path(input_path)
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            yield RefinementTrace.from_dict(data)


def load_jsonl(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load generic JSONL file.
    
    Args:
        input_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    records = []
    input_path = Path(input_path)
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            records.append(data)
    
    return records


def save_jsonl(
    records: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """
    Save list of dictionaries to JSONL file.
    
    Args:
        records: List of dictionaries to save
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")


def hash_case_text(case_text: str) -> str:
    """
    Generate a hash of case text for alignment purposes.
    
    Args:
        case_text: The case presentation text
        
    Returns:
        MD5 hash string (first 16 characters)
    """
    # Normalize whitespace for consistent hashing
    normalized = " ".join(case_text.split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


def save_summary_report(
    report: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Save summary report to JSON file.
    
    Args:
        report: Report dictionary
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def load_summary_report(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load summary report from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Report dictionary
    """
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


class CSVExporter:
    """Export refinement results to CSV format compatible with baseline."""
    
    @staticmethod
    def export_for_bertscore(
        traces: List[RefinementTrace],
        output_path: Union[str, Path],
    ) -> None:
        """
        Export traces to CSV format compatible with baseline BERTScore evaluation.
        
        Creates a CSV with columns:
        - Case presentation
        - True diagnosis
        - Generated diagnosis (extracted from refined response)
        
        Args:
            traces: List of RefinementTrace objects
            output_path: Path to output CSV file
        """
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Case presentation",
                "True diagnosis",
                "Generated diagnosis",
            ])
            writer.writeheader()
            
            for trace in traces:
                writer.writerow({
                    "Case presentation": trace.case_text,
                    "True diagnosis": trace.true_diagnosis,
                    "Generated diagnosis": trace.extracted_final_diagnosis,
                })
    
    @staticmethod
    def export_with_metrics(
        traces: List[RefinementTrace],
        bertscore_f1: Optional[List[float]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Export traces to CSV with BERTScore F1 and additional metrics.
        
        Args:
            traces: List of RefinementTrace objects
            bertscore_f1: Optional list of BERTScore F1 values
            output_path: Path to output CSV file
        """
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            "Case presentation",
            "True diagnosis",
            "Generated diagnosis",
            "BERTScore F1",
            "Is Compliant",
            "Iterations to Compliance",
            "Clinical Quality Score",
            "Edit Distance Total",
            "Edit Ratio Total",
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, trace in enumerate(traces):
                row = {
                    "Case presentation": trace.case_text,
                    "True diagnosis": trace.true_diagnosis,
                    "Generated diagnosis": trace.extracted_final_diagnosis,
                    "BERTScore F1": bertscore_f1[i] if bertscore_f1 else None,
                    "Is Compliant": trace.is_compliant,
                    "Iterations to Compliance": trace.iterations_to_compliance,
                    "Clinical Quality Score": trace.clinical_quality_score,
                    "Edit Distance Total": trace.minimality_metrics.get("edit_distance_total"),
                    "Edit Ratio Total": trace.minimality_metrics.get("edit_ratio_total"),
                }
                writer.writerow(row)
