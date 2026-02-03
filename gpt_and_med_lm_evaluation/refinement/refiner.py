"""
Iterative Adversarial Refinement with Checklist Enforcement.

Main refiner class implementing the Generator -> Critic -> Editor loop.
"""

import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import logging

from openai import OpenAI

from .schema import (
    DiagnosticResponse,
    CriticResult,
    RefinementTrace,
    IterationLog,
    ChecklistConfig,
    parse_diagnostic_response,
    parse_critic_result,
    validate_diagnostic_response,
)
from .metrics import (
    compute_minimality_metrics,
    compute_ccr_for_case,
)


logger = logging.getLogger(__name__)


def load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompts_dir = Path(__file__).parent / "prompts"
    template_path = prompts_dir / f"{name}.md"
    
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class RefinerConfig:
    """Configuration for the IterativeRefiner."""
    
    generator_model: str = "gpt-4o"
    critic_model: str = "gpt-4o"
    editor_model: str = "gpt-4o"
    max_iterations: int = 3
    clinical_quality_threshold: int = 3
    retry_attempts: int = 3
    retry_delay: float = 60.0
    api_delay: float = 1.0
    temperature: float = 0.0
    checklist_config_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_model": self.generator_model,
            "critic_model": self.critic_model,
            "editor_model": self.editor_model,
            "max_iterations": self.max_iterations,
            "clinical_quality_threshold": self.clinical_quality_threshold,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "api_delay": self.api_delay,
            "temperature": self.temperature,
        }


class IterativeRefiner:
    """
    Implements iterative adversarial refinement with checklist enforcement.
    
    Algorithm:
    1. Generate initial response (Generator)
    2. For each iteration up to max_iterations:
       a. Critique the response (Critic)
       b. If joint compliance met → return
       c. Else → Edit response (Editor)
    3. If not compliant by max iterations → return best attempt
    """
    
    def __init__(
        self,
        client: OpenAI,
        config: Optional[RefinerConfig] = None,
    ):
        """
        Initialize the IterativeRefiner.
        
        Args:
            client: OpenAI client instance
            config: Optional RefinerConfig (uses defaults if None)
        """
        self.client = client
        self.config = config or RefinerConfig()
        
        # Load checklist configuration
        self.checklist_config = ChecklistConfig.load(
            self.config.checklist_config_path
        )
        
        # Load prompt templates
        self._generator_template = load_prompt_template("generator")
        self._critic_template = load_prompt_template("critic")
        self._editor_template = load_prompt_template("editor")
    
    def _call_api(
        self,
        model: str,
        prompt: str,
        parse_fn: Callable[[str], Any],
    ) -> Tuple[Any, str]:
        """
        Call OpenAI API with retry logic and parsing.
        
        Args:
            model: Model to use
            prompt: Prompt to send
            parse_fn: Function to parse the response
            
        Returns:
            Tuple of (parsed_result, raw_response)
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        raw_response = ""
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                )
                raw_response = response.choices[0].message.content.strip()
                
                # Parse the response
                parsed = parse_fn(raw_response)
                
                # Add delay between calls
                time.sleep(self.config.api_delay)
                
                return parsed, raw_response
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"API call attempt {attempt + 1}/{self.config.retry_attempts} failed: {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        
        raise last_error
    
    def generate(self, case_text: str) -> Tuple[DiagnosticResponse, str]:
        """
        Generate initial diagnostic response.
        
        Args:
            case_text: The case presentation text
            
        Returns:
            Tuple of (DiagnosticResponse, raw_response)
        """
        prompt = self._generator_template.replace("{case_text}", case_text)
        
        return self._call_api(
            model=self.config.generator_model,
            prompt=prompt,
            parse_fn=parse_diagnostic_response,
        )
    
    def critique(
        self,
        case_text: str,
        response: DiagnosticResponse,
    ) -> Tuple[CriticResult, str]:
        """
        Critique a diagnostic response.
        
        Args:
            case_text: The case presentation text
            response: Current DiagnosticResponse
            
        Returns:
            Tuple of (CriticResult, raw_response)
        """
        prompt = self._critic_template.replace(
            "{case_text}", case_text
        ).replace(
            "{current_response}", response.to_json()
        )
        
        return self._call_api(
            model=self.config.critic_model,
            prompt=prompt,
            parse_fn=parse_critic_result,
        )
    
    def edit(
        self,
        case_text: str,
        response: DiagnosticResponse,
        edit_plan: List[str],
    ) -> Tuple[DiagnosticResponse, str]:
        """
        Edit a diagnostic response based on critic feedback.
        
        Args:
            case_text: The case presentation text
            response: Previous DiagnosticResponse
            edit_plan: List of edit actions from Critic
            
        Returns:
            Tuple of (DiagnosticResponse, raw_response)
        """
        edit_plan_text = "\n".join(f"- {action}" for action in edit_plan)
        
        prompt = self._editor_template.replace(
            "{case_text}", case_text
        ).replace(
            "{previous_response}", response.to_json()
        ).replace(
            "{edit_plan}", edit_plan_text
        )
        
        return self._call_api(
            model=self.config.editor_model,
            prompt=prompt,
            parse_fn=parse_diagnostic_response,
        )
    
    def is_jointly_compliant(self, critic_result: CriticResult) -> bool:
        """
        Check if response meets joint compliance criteria.
        
        Args:
            critic_result: CriticResult from the Critic
            
        Returns:
            True if all criteria met
        """
        return critic_result.is_compliant(self.config.clinical_quality_threshold)
    
    def refine(
        self,
        case_text: str,
        case_id: str = "",
        true_diagnosis: str = "",
    ) -> RefinementTrace:
        """
        Run the full iterative refinement process for a single case.
        
        Args:
            case_text: The case presentation text
            case_id: Optional case identifier
            true_diagnosis: Optional ground truth diagnosis
            
        Returns:
            RefinementTrace with full iteration history and metrics
        """
        iterations: List[IterationLog] = []
        responses: List[DiagnosticResponse] = []
        best_response: Optional[DiagnosticResponse] = None
        best_critic: Optional[CriticResult] = None
        best_quality = -1
        best_failed_count = 999
        
        # Step 1: Generate initial response
        try:
            draft, _ = self.generate(case_text)
        except Exception as e:
            logger.error(f"Generation failed for case {case_id}: {e}")
            # Return a minimal trace with error
            return self._create_error_trace(
                case_id, case_text, true_diagnosis, str(e)
            )
        
        responses.append(draft)
        
        # Step 2: Iterate
        iterations_to_compliance: Optional[int] = None
        is_compliant = False
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Critique current draft
            try:
                critic_result, _ = self.critique(case_text, draft)
            except Exception as e:
                logger.warning(f"Critique failed at iteration {iteration}: {e}")
                # Create a failing critic result
                critic_result = self._create_format_failure_critic()
            
            # Log this iteration
            iterations.append(IterationLog(
                iteration=iteration,
                response=draft,
                critic_result=critic_result,
            ))
            
            # Track best response
            quality = critic_result.clinical_quality.score
            failed_count = len(critic_result.get_failed_items())
            
            if (quality > best_quality) or (
                quality == best_quality and failed_count < best_failed_count
            ):
                best_quality = quality
                best_failed_count = failed_count
                best_response = draft
                best_critic = critic_result
            
            # Check joint compliance
            if self.is_jointly_compliant(critic_result):
                iterations_to_compliance = iteration
                is_compliant = True
                break
            
            # Edit the response
            if iteration < self.config.max_iterations:
                try:
                    draft, _ = self.edit(
                        case_text, draft, critic_result.edit_plan
                    )
                    responses.append(draft)
                except Exception as e:
                    logger.warning(f"Edit failed at iteration {iteration}: {e}")
                    # Keep the current draft
        
        # Use best response if not compliant
        if not is_compliant and best_response is not None:
            draft = best_response
        
        # Compute metrics
        minimality_metrics = compute_minimality_metrics(responses)
        
        # Build checklist pass map
        final_critic = best_critic if best_critic else (
            iterations[-1].critic_result if iterations else None
        )
        
        checklist_pass_map = {}
        clinical_quality_score = None
        hard_fail = False
        
        if final_critic:
            checklist_pass_map = {
                item.item_id: item.passed
                for item in final_critic.checklist
            }
            clinical_quality_score = final_critic.clinical_quality.score
            hard_fail = final_critic.hard_fail.failed
        
        # Build trace
        trace = RefinementTrace(
            case_id=case_id,
            case_text=case_text,
            true_diagnosis=true_diagnosis,
            final_response=draft,
            extracted_final_diagnosis=draft.final_diagnosis,
            iterations_to_compliance=iterations_to_compliance,
            is_compliant=is_compliant,
            iterations=iterations,
            minimality_metrics=minimality_metrics.to_dict(),
            checklist_pass_map=checklist_pass_map,
            clinical_quality_score=clinical_quality_score,
            hard_fail=hard_fail,
        )
        
        return trace
    
    def _create_error_trace(
        self,
        case_id: str,
        case_text: str,
        true_diagnosis: str,
        error_message: str,
    ) -> RefinementTrace:
        """Create a trace for a failed case."""
        error_response = DiagnosticResponse(
            final_diagnosis=f"[ERROR: {error_message}]",
            next_steps=[],
        )
        
        return RefinementTrace(
            case_id=case_id,
            case_text=case_text,
            true_diagnosis=true_diagnosis,
            final_response=error_response,
            extracted_final_diagnosis=error_response.final_diagnosis,
            iterations_to_compliance=None,
            is_compliant=False,
            iterations=[],
            minimality_metrics={},
            checklist_pass_map={},
            clinical_quality_score=None,
            hard_fail=True,
        )
    
    def _create_format_failure_critic(self) -> CriticResult:
        """Create a critic result indicating format/parsing failure."""
        from .schema import ChecklistItemResult, ClinicalQuality, HardFail
        
        # Create failing checklist items
        items = [
            ChecklistItemResult(
                item_id=f"C{i}",
                passed=False,
                rationale="Unable to evaluate due to format error",
                suggested_fix="Fix response format to valid JSON",
            )
            for i in range(1, 9)
        ]
        
        return CriticResult(
            checklist=items,
            clinical_quality=ClinicalQuality(
                score=0,
                rationale="Cannot evaluate due to format error",
            ),
            hard_fail=HardFail(
                failed=True,
                reason="Response format could not be parsed",
            ),
            edit_plan=["Fix response format to valid JSON with required fields"],
        )


def create_refiner(
    api_key: Optional[str] = None,
    config: Optional[RefinerConfig] = None,
) -> IterativeRefiner:
    """
    Factory function to create an IterativeRefiner with OpenAI client.
    
    Args:
        api_key: Optional API key (uses environment variable if None)
        config: Optional RefinerConfig
        
    Returns:
        Configured IterativeRefiner
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    client = OpenAI(api_key=api_key)
    
    return IterativeRefiner(client=client, config=config)


class BatchRefiner:
    """Helper class for processing batches of cases with progress tracking."""
    
    def __init__(
        self,
        refiner: IterativeRefiner,
        batch_delay: float = 10.0,
    ):
        """
        Initialize BatchRefiner.
        
        Args:
            refiner: IterativeRefiner instance
            batch_delay: Delay between batches in seconds
        """
        self.refiner = refiner
        self.batch_delay = batch_delay
    
    def process_batch(
        self,
        cases: List[Dict[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[RefinementTrace]:
        """
        Process a batch of cases.
        
        Args:
            cases: List of dicts with 'case_text', 'case_id', 'true_diagnosis'
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of RefinementTrace objects
        """
        traces = []
        total = len(cases)
        
        for i, case in enumerate(cases):
            if progress_callback:
                progress_callback(i, total)
            
            trace = self.refiner.refine(
                case_text=case.get("case_text", ""),
                case_id=case.get("case_id", str(i)),
                true_diagnosis=case.get("true_diagnosis", ""),
            )
            traces.append(trace)
        
        return traces
    
    def process_batches(
        self,
        all_cases: List[Dict[str, str]],
        batch_size: int = 250,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[RefinementTrace]:
        """
        Process multiple batches of cases.
        
        Args:
            all_cases: List of all cases
            batch_size: Size of each batch
            progress_callback: Optional callback(batch_num, case_num, total)
            
        Returns:
            List of all RefinementTrace objects
        """
        all_traces = []
        n_batches = (len(all_cases) + batch_size - 1) // batch_size
        
        for batch_num in range(n_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_cases))
            batch = all_cases[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{n_batches}")
            
            for i, case in enumerate(batch):
                if progress_callback:
                    progress_callback(batch_num + 1, i + 1, len(batch))
                
                trace = self.refiner.refine(
                    case_text=case.get("case_text", ""),
                    case_id=case.get("case_id", str(start_idx + i)),
                    true_diagnosis=case.get("true_diagnosis", ""),
                )
                all_traces.append(trace)
            
            # Delay between batches
            if batch_num < n_batches - 1:
                time.sleep(self.batch_delay)
        
        return all_traces
