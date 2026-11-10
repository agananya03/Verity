"""
Quota Ledger and Free-Tier Budget Tracking for Verity RAG Evaluation.

Tracks API requests per model per day using America/Los_Angeles dates
(since Google Gemini API quotas reset at Pacific midnight).

Features:
- Configurable limits via backend/eval/quota_limits.json.
- Persistent daily ledger at backend/eval/results/quota_ledger.json.
- Next quota reset calculation in Indian Standard Time (IST).
- Graceful stopping with exit code 2 when model budget hits 0 or --max-calls reached.
"""

import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_LIMITS_PATH = EVAL_DIR / "quota_limits.json"
DEFAULT_LEDGER_PATH = RESULTS_DIR / "quota_ledger.json"

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
IST_TZ = ZoneInfo("Asia/Kolkata")

# Safety factor applied to configured quota limits.
# Note: The ledger can undercount SDK-internal retries (e.g. transient 503 retries or
# LangChain / google-genai client internal retries), so applying a 0.8 safety factor
# provides a 20% margin to prevent unexpected quota exhaustion.
QUOTA_SAFETY_FACTOR = 0.8


def get_pacific_today() -> str:
    """Returns today's date in America/Los_Angeles timezone (YYYY-MM-DD)."""
    return datetime.datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")


def get_next_reset_time_ist() -> str:
    """
    Returns the next Pacific midnight (00:00:00 America/Los_Angeles)
    converted to Indian Standard Time (Asia/Kolkata), formatted nicely.
    """
    now_la = datetime.datetime.now(PACIFIC_TZ)
    midnight_la = (now_la + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    midnight_ist = midnight_la.astimezone(IST_TZ)
    return midnight_ist.strftime("%Y-%m-%d %I:%M:%S %p IST")


class QuotaLedger:
    """
    Manages daily LLM quota tracking and budget enforcement across Gemini models.
    """

    def __init__(
        self,
        limits_path: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
        max_calls: Optional[int] = None,
    ):
        self.limits_path = limits_path or DEFAULT_LIMITS_PATH
        self.ledger_path = ledger_path or DEFAULT_LEDGER_PATH
        self.max_calls = max_calls
        self.run_calls = 0

    def load_limits(self) -> Dict[str, int]:
        """Loads model daily request limits from quota_limits.json and applies 0.8 safety factor."""
        if not self.limits_path.exists():
            return {
                "gemini-3.5-flash": int(1500 * QUOTA_SAFETY_FACTOR),
                "gemini-3.5-flash-lite": int(1500 * QUOTA_SAFETY_FACTOR),
                "gemini-2.5-flash": int(1500 * QUOTA_SAFETY_FACTOR),
                "gemini-2.5-flash-lite": int(1500 * QUOTA_SAFETY_FACTOR),
                "gemini-2.5-pro": int(50 * QUOTA_SAFETY_FACTOR),
                "gemini-3.5-pro": int(50 * QUOTA_SAFETY_FACTOR),
            }
        try:
            with open(self.limits_path, "r", encoding="utf-8") as f:
                raw_limits = json.load(f)
                return {
                    model: int(limit * QUOTA_SAFETY_FACTOR)
                    for model, limit in raw_limits.items()
                }
        except Exception as e:
            logger.warning(f"Failed to read quota limits from {self.limits_path}: {e}")
            return {}

    def load_ledger(self) -> Dict[str, Dict[str, int]]:
        """Loads daily call counts from quota_ledger.json."""
        if not self.ledger_path.exists():
            return {}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("dates", data) if "dates" in data else data
        except Exception as e:
            logger.warning(f"Failed to read quota ledger from {self.ledger_path}: {e}")
            return {}

    def save_ledger(self, ledger_data: Dict[str, Any]) -> None:
        """Saves daily call counts to quota_ledger.json safely."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write quota ledger to {self.ledger_path}: {e}")

    def get_limit(self, model: str) -> int:
        """Returns the daily request limit for the given model with safety factor applied."""
        limits = self.load_limits()
        if model in limits:
            return limits[model]
        # Heuristic fallback for unlisted models with 0.8 safety factor
        raw = 50 if "pro" in model.lower() else 1500
        return int(raw * QUOTA_SAFETY_FACTOR)

    def get_used_today(self, model: str, date_str: Optional[str] = None) -> int:
        """Returns the number of calls made to model today (in Pacific time)."""
        date_key = date_str or get_pacific_today()
        ledger = self.load_ledger()
        date_data = ledger.get(date_key, {})
        return date_data.get(model, 0)

    def get_remaining_budget(self, model: str, date_str: Optional[str] = None) -> int:
        """Returns remaining daily requests for the model today."""
        limit = self.get_limit(model)
        used = self.get_used_today(model, date_str)
        return max(0, limit - used)

    def record_call(self, model: str, count: int = 1) -> int:
        """
        Increments the call count for model on today's Pacific date.
        Returns the new total calls made today.
        """
        date_key = get_pacific_today()
        ledger = self.load_ledger()
        if date_key not in ledger:
            ledger[date_key] = {}
        curr = ledger[date_key].get(model, 0)
        new_val = curr + count
        ledger[date_key][model] = new_val
        self.save_ledger(ledger)

        self.run_calls += count
        return new_val

    def check_budget_or_stop(
        self,
        model: str,
        save_partial_callback: Optional[Callable[[], None]] = None,
        next_index: Optional[int] = None,
    ) -> None:
        """
        Verifies that model still has quota budget and --max-calls hasn't been exceeded.
        If budget hits 0 or max-calls is exceeded:
          1. Invokes save_partial_callback() to persist progress.
          2. Prints loud summary with reset time in IST.
          3. Exits cleanly with exit code 2.
        """
        # Check run-level max-calls limit
        if self.max_calls is not None and self.run_calls >= self.max_calls:
            if save_partial_callback:
                save_partial_callback()
            reset_ist = get_next_reset_time_ist()
            print("\n" + "=" * 78)
            print("MAX-CALLS LIMIT REACHED: Evaluation stopped gracefully.")
            print(f"Calls made in this run: {self.run_calls} (limit: {self.max_calls})")
            if next_index is not None:
                print(f"Resume with:            --start-from {next_index}")
            print(f"Quota resets at:        {reset_ist}")
            print("=" * 78 + "\n")
            sys.exit(2)

        # Check model daily budget
        rem = self.get_remaining_budget(model)
        if rem <= 0:
            if save_partial_callback:
                save_partial_callback()
            limit = self.get_limit(model)
            used = self.get_used_today(model)
            reset_ist = get_next_reset_time_ist()

            print("\n" + "!" * 78)
            print("QUOTA BUDGET EXHAUSTED: Evaluation stopped gracefully.")
            print(f"Model:          '{model}' has 0 remaining requests today.")
            print(f"Daily Limit:    {limit} requests")
            print(f"Used Today:     {used} requests (America/Los_Angeles date: {get_pacific_today()})")
            print(f"Quota Resets:   {reset_ist}")
            if next_index is not None:
                print(f"Resume with:    --start-from {next_index}")
            print("!" * 78 + "\n")
            logger.warning(
                f"Graceful stop: Model '{model}' daily budget exhausted. Resets at {reset_ist}."
            )
            sys.exit(2)


# Global singleton instance
global_quota_ledger = QuotaLedger()

