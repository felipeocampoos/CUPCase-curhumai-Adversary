"""
Iterative Adversarial Refinement with Checklist Enforcement.

Main refiner class implementing the Generator -> Critic -> Editor loop.
"""

import time
import json
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from openai import OpenAI


class JudgeProvider(str, Enum):
    """Supported LLM providers for judge/evaluation."""
    
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OPENAI_COMPATIBLE = "openai_compatible"
    HUGGINGFACE_LOCAL = "huggingface_local"
    
    @property
    def base_url(self) -> Optional[str]:
        """Get the API base URL for the provider."""
        import os

        if self == JudgeProvider.DEEPSEEK:
            return "https://api.deepseek.com"
        if self == JudgeProvider.OPENAI_COMPATIBLE:
            return os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        return None  # OpenAI uses default
    
    @property
    def default_model(self) -> str:
        """Get the default model for the provider."""
        import os

        if self == JudgeProvider.DEEPSEEK:
            return "deepseek-chat"
        if self == JudgeProvider.OPENAI_COMPATIBLE:
            return os.environ.get("OPENAI_COMPATIBLE_MODEL", "openai-compatible-model")
        if self == JudgeProvider.HUGGINGFACE_LOCAL:
            return os.environ.get("HUGGINGFACE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        return "gpt-4o"
    
    @property
    def env_var(self) -> str:
        """Get the environment variable name for API key."""
        if self == JudgeProvider.DEEPSEEK:
            return "DEEPSEEK_API_KEY"
        if self == JudgeProvider.OPENAI_COMPATIBLE:
            return "OPENAI_COMPATIBLE_API_KEY"
        if self == JudgeProvider.HUGGINGFACE_LOCAL:
            return "HF_TOKEN"
        return "OPENAI_API_KEY"


class _HFLocalMessage:
    def __init__(self, content: str):
        self.content = content


class _HFLocalChoice:
    def __init__(self, content: str):
        self.message = _HFLocalMessage(content)


class _HFLocalCompletionResponse:
    def __init__(self, content: str):
        self.choices = [_HFLocalChoice(content)]


class _HFLocalModelsResponse:
    def __init__(self, model_id: str):
        self.data = [type("Model", (), {"id": model_id})()]


class HuggingFaceLocalClient:
    """Minimal OpenAI-like client backed by local Hugging Face generation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._loaded_model_id: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self.chat = type("Chat", (), {"completions": self})()
        self.models = type("Models", (), {"list": self._list_models})()

    def _list_models(self):
        model_id = self._loaded_model_id or JudgeProvider.HUGGINGFACE_LOCAL.default_model
        return _HFLocalModelsResponse(model_id)

    def _ensure_loaded(self, model_id: str):
        if self._loaded_model_id == model_id and self._model is not None and self._tokenizer is not None:
            return

        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path

        cache_root = Path(os.environ.get("HF_HOME", "/tmp/cupcase_hf_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root)
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))

        token = self.api_key or None
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            cache_dir=str(cache_root),
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            cache_dir=str(cache_root),
        )
        self._model.eval()
        self._loaded_model_id = model_id

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, **kwargs):
        import torch

        self._ensure_loaded(model)

        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=int(kwargs.get("max_tokens", 96)),
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _HFLocalCompletionResponse(content)

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
    similarity_threshold: float = 0.65
    disclosure_fraction: float = 0.2
    early_confidence_threshold: float = 0.8
    revision_instability_threshold: float = 0.5
    checklist_config_path: Optional[Path] = None
    curiosity_threshold: int = 0
    humility_threshold: int = 0
    provider: JudgeProvider = JudgeProvider.OPENAI

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
            "similarity_threshold": self.similarity_threshold,
            "disclosure_fraction": self.disclosure_fraction,
            "early_confidence_threshold": self.early_confidence_threshold,
            "revision_instability_threshold": self.revision_instability_threshold,
            "curiosity_threshold": self.curiosity_threshold,
            "humility_threshold": self.humility_threshold,
            "provider": self.provider.value,
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
    variant_name: str = "baseline"
    
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
        if not critic_result.is_compliant(self.config.clinical_quality_threshold):
            return False

        if self.config.curiosity_threshold > 0:
            score = critic_result.curiosity_score
            if score is None or score < self.config.curiosity_threshold:
                return False

        if self.config.humility_threshold > 0:
            score = critic_result.humility_score
            if score is None or score < self.config.humility_threshold:
                return False

        return True

    def _get_case_variant_metadata(self) -> Dict[str, Any]:
        """
        Return variant-specific metadata for the current case.

        Subclasses can override this to attach routing decisions or other
        variant artifacts to the trace.
        """
        return {}
    
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

        variant_initial_response = deepcopy(draft)
        variant_stage_metadata = dict(self._get_case_variant_metadata())
        responses.append(draft)
        
        # Step 2: Iterate
        iterations_to_compliance: Optional[int] = None
        is_compliant = False
        hard_fail_any_iteration = False
        first_failure_iteration: Optional[int] = None
        
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

            if critic_result.hard_fail.failed:
                hard_fail_any_iteration = True
            
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
            is_jointly_compliant = self.is_jointly_compliant(critic_result)
            if not is_jointly_compliant and first_failure_iteration is None:
                first_failure_iteration = iteration

            if is_jointly_compliant:
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
        curiosity_score = None
        humility_score = None

        if final_critic:
            checklist_pass_map = {
                item.item_id: item.passed
                for item in final_critic.checklist
            }
            clinical_quality_score = final_critic.clinical_quality.score
            hard_fail = final_critic.hard_fail.failed
            curiosity_score = final_critic.curiosity_score
            humility_score = final_critic.humility_score
        
        # Build trace
        diagnosis_trajectory = [response.final_diagnosis for response in responses]
        editor_recovered_case = bool(
            is_compliant
            and first_failure_iteration is not None
            and iterations_to_compliance is not None
            and iterations_to_compliance > first_failure_iteration
        )

        trace = RefinementTrace(
            case_id=case_id,
            case_text=case_text,
            true_diagnosis=true_diagnosis,
            variant_initial_response=variant_initial_response,
            variant_initial_diagnosis=variant_initial_response.final_diagnosis,
            final_response=draft,
            extracted_final_diagnosis=draft.final_diagnosis,
            iterations_to_compliance=iterations_to_compliance,
            is_compliant=is_compliant,
            iterations=iterations,
            minimality_metrics=minimality_metrics.to_dict(),
            checklist_pass_map=checklist_pass_map,
            clinical_quality_score=clinical_quality_score,
            hard_fail=hard_fail,
            hard_fail_any_iteration=hard_fail_any_iteration,
            first_failure_iteration=first_failure_iteration,
            editor_recovered_case=editor_recovered_case,
            curiosity_score=curiosity_score,
            humility_score=humility_score,
            variant_name=self.variant_name,
            variant_metadata=dict(self._get_case_variant_metadata()),
            variant_stage_metadata=variant_stage_metadata,
            diagnosis_trajectory=diagnosis_trajectory,
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
            variant_initial_response=error_response,
            variant_initial_diagnosis=error_response.final_diagnosis,
            final_response=error_response,
            extracted_final_diagnosis=error_response.final_diagnosis,
            iterations_to_compliance=None,
            is_compliant=False,
            iterations=[],
            minimality_metrics={},
            checklist_pass_map={},
            clinical_quality_score=None,
            hard_fail=True,
            hard_fail_any_iteration=True,
            first_failure_iteration=1,
            editor_recovered_case=False,
            variant_name=self.variant_name,
            variant_metadata=dict(self._get_case_variant_metadata()),
            variant_stage_metadata=dict(self._get_case_variant_metadata()),
            diagnosis_trajectory=[error_response.final_diagnosis],
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
            edit_plan=["[GENERAL] Fix response format to valid JSON with required fields"],
            curiosity_score=0,
            humility_score=0,
        )


def create_client(
    provider: JudgeProvider = JudgeProvider.OPENAI,
    api_key: Optional[str] = None,
) -> OpenAI:
    """
    Factory function to create an OpenAI-compatible client for the specified provider.
    
    Args:
        provider: The LLM provider to use (openai, deepseek, openai_compatible, huggingface_local)
        api_key: Optional API key (uses environment variable if None)
        
    Returns:
        OpenAI client configured for the provider
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if api_key is None:
        api_key = os.environ.get(provider.env_var)

    if provider == JudgeProvider.HUGGINGFACE_LOCAL:
        return HuggingFaceLocalClient(api_key=api_key)

    if provider == JudgeProvider.OPENAI_COMPATIBLE and not api_key:
        # Local vLLM / llama.cpp style servers commonly do not require auth.
        api_key = "dummy"

    base_url = provider.base_url
    if provider == JudgeProvider.OPENAI_COMPATIBLE and not base_url:
        raise ValueError(
            "OPENAI_COMPATIBLE_BASE_URL must be set when provider=openai_compatible"
        )

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def create_refiner(
    api_key: Optional[str] = None,
    config: Optional[RefinerConfig] = None,
    provider: Optional[JudgeProvider] = None,
) -> IterativeRefiner:
    """
    Factory function to create an IterativeRefiner with OpenAI-compatible client.
    
    Args:
        api_key: Optional API key (uses environment variable if None)
        config: Optional RefinerConfig
        provider: Optional provider (defaults to config.provider or OPENAI)
        
    Returns:
        Configured IterativeRefiner
    """
    if config is None:
        config = RefinerConfig()
    
    # Determine provider: explicit arg > config > default
    if provider is None:
        provider = config.provider
    
    client = create_client(provider=provider, api_key=api_key)
    
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
