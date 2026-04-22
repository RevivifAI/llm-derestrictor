"""Refusal rate evaluation using log-probability based detection.

This module provides a standalone scanner for evaluating model refusal rates
on a set of prompts. It uses the shared LogLikelihoodRefusalDetector for
consistent detection across the codebase.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from derestrictor.eval.detector import (
    LogLikelihoodRefusalDetector,
    RefusalDetectorConfig,
)
from derestrictor.models.utils import load_model_and_tokenizer


class RefusalScanner:
    """Standalone scanner for evaluating refusal rates on prompt files.

    Loads its own model and uses LogLikelihoodRefusalDetector for detection.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
        dtype: str = "auto",
        debug: bool = False,
        threshold: float = -7.0,
    ):
        self.device = device
        self.batch_size = batch_size
        self.debug = debug
        self.threshold = threshold

        # Determine dtype - bfloat16 is more stable for many models (avoids NaN)
        if dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"Loading model: {model_name}...")
        print(f"Using dtype: {torch_dtype}")

        # Use shared loader for VL model support
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name,
            device=device,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
        # Ensure padding works for batching
        self.tokenizer.padding_side = "left"
        self.model.eval()

        # Create the shared detector
        config = RefusalDetectorConfig(threshold=threshold)
        self._detector = LogLikelihoodRefusalDetector(self.model, self.tokenizer, config)

    def format_prompt(self, prompt: str) -> str:
        """Format a prompt using the model's chat template."""
        return self._detector.format_prompt(prompt)

    def get_log_prob(self, prompts: list[str], target_string: str) -> torch.Tensor:
        """Calculate the normalized log-probability of ``target_string``.

        Prompts must already be formatted with the chat template. Uses batched
        processing for efficiency.
        """
        return self._detector.get_log_prob(prompts, target_string)

    def scan_split(
        self,
        split: str,
        output_file: str,
        limit: int | None = None,
        model_name: str | None = None,
        save_csv: bool = True,
        threshold: float | None = None,
    ):
        """Scan prompts from a RevivifAI/derestriction split and score refusals."""
        from derestrictor.data.loader import load_split

        print(f"Reading prompts from RevivifAI/derestriction split={split}...")
        prompts = load_split(split)

        if limit is not None:
            prompts = prompts[:limit]

        if output_file == "refusal_results.csv" and model_name:
            output_dir = Path("eval_abliterated")
            output_dir.mkdir(exist_ok=True)

            clean_name = Path(model_name).name.replace("/", "_").replace("\\", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            limit_str = f"_n{limit}" if limit else "_all"
            output_file = output_dir / f"refusal_eval_{clean_name}{limit_str}_{timestamp}.csv"

        results = []

        print(f"Scanning {len(prompts)} prompts...")

        # Debug: show formatted prompt example
        if self.debug and prompts:
            example_formatted = self.format_prompt(prompts[0])
            print(f"Example formatted prompt:\n{example_formatted!r}")

        # Save and restore threshold if overridden
        original_threshold = self._detector.config.threshold
        try:
            if threshold is not None:
                self._detector.config.threshold = threshold

            for i in tqdm(range(0, len(prompts), self.batch_size)):
                batch_prompts = prompts[i : i + self.batch_size]

                try:
                    # Use the detector's batch detection with scores
                    batch_results = self._detector.detect_refusal_with_scores(batch_prompts)

                    # Debug first batch
                    if self.debug and i == 0:
                        print(f"First batch results: {batch_results}")
                        print(f"Refusal threshold: {self._detector.config.threshold}")

                    for j, prompt in enumerate(batch_prompts):
                        is_refusal, r_score = batch_results[j]

                        results.append({"prompt": prompt, "is_refusal": is_refusal, "refusal_score": round(r_score, 4)})
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    import traceback

                    traceback.print_exc()
                    # Add failed results
                    results.extend(
                        {"prompt": prompt, "is_refusal": None, "refusal_score": None} for prompt in batch_prompts
                    )
        finally:
            self._detector.config.threshold = original_threshold

        # 3. Save Results
        df = pd.DataFrame(results)
        if save_csv:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        print(f"Refusal Rate: {df['is_refusal'].mean():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Refusal Scanner")
    parser.add_argument(
        "--split",
        type=str,
        default="derestrict",
        choices=["restrict", "derestrict", "allow"],
        help="RevivifAI/derestriction split to scan (default: derestrict).",
    )
    parser.add_argument("--output", type=str, default="refusal_results.csv", help="Output CSV path")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it", help="Hugging Face model ID")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to scan")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype (auto uses bfloat16 if supported)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    parser.add_argument(
        "--threshold", type=float, default=-7.0, help="Refusal threshold (log prob above this = refusing)"
    )

    args = parser.parse_args()

    scanner = RefusalScanner(
        args.model,
        batch_size=args.batch_size,
        dtype=args.dtype,
        debug=args.debug,
        threshold=args.threshold,
    )
    scanner.scan_split(
        args.split,
        args.output,
        limit=args.limit,
        model_name=args.model,
        save_csv=not args.no_csv,
        threshold=args.threshold,
    )
