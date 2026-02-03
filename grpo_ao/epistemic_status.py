"""Handle epistemic status format for oracle outputs.

Format: [epistemic status: XX] <answer>

Where XX is an integer from 0 to 100.

Examples:
    [epistemic status: 85] The text discusses a long-distance relationship...
    [epistemic status: 30] This appears to be about machine learning...
    [epistemic status: 10] I cannot determine the topic from these activations.
"""

import re
from dataclasses import dataclass


@dataclass
class OracleOutput:
    """Parsed oracle output with confidence and answer."""

    confidence: int  # 0-100
    answer: str
    raw: str
    parse_success: bool

    @property
    def confidence_normalized(self) -> float:
        """Return confidence as 0-1 float for reward computation."""
        return self.confidence / 100.0


# Regex to parse epistemic status (now 0-100 integer)
EPISTEMIC_PATTERN = re.compile(
    r"^\s*\[epistemic status:\s*(\d+)\s*\]\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


def parse_oracle_output(raw_output: str, default_confidence: int = -1) -> OracleOutput:
    """Parse oracle output into confidence and answer.

    If parsing fails, returns default_confidence (-1) which signals
    malformed output for a penalty in reward computation.

    Args:
        raw_output: Raw string from oracle generation
        default_confidence: Confidence to use if parsing fails (0-100)

    Returns:
        OracleOutput with parsed or default values
    """
    raw_output = raw_output.strip()

    match = EPISTEMIC_PATTERN.match(raw_output)
    if match:
        try:
            confidence = int(match.group(1))
            # Clamp to [0, 100]
            confidence = max(0, min(100, confidence))
            answer = match.group(2).strip()
            return OracleOutput(
                confidence=confidence,
                answer=answer,
                raw=raw_output,
                parse_success=True
            )
        except ValueError:
            pass

    # Parsing failed - return default
    return OracleOutput(
        confidence=default_confidence,
        answer=raw_output,
        raw=raw_output,
        parse_success=False
    )


def format_epistemic_output(confidence: int, answer: str) -> str:
    """Format answer with epistemic status prefix.

    Args:
        confidence: Confidence level (0-100)
        answer: The answer text

    Returns:
        Formatted string with epistemic status prefix
    """
    return f"[epistemic status: {confidence}] {answer}"


# System prompt to teach the oracle the format
ORACLE_SYSTEM_PROMPT = """Respond with: [epistemic status: XX] Answer

XX is your confidence 0-100. BE PRECISE - don't always say 50!

Use the FULL range based on signal clarity:
- 95: crystal clear signal, absolutely certain
- 80: strong signal, very confident
- 60: decent signal, somewhat confident
- 40: weak signal, somewhat uncertain

etc...

IMPORTANT: if you are certain that you DON'T KNOW: don't get confused!

[epistemic status: 100] I can't answer this question given these activations.

is a valid answer!

Examples:
[epistemic status: 65] Yes
[epistemic status: 68] No
[epistemic status: 87] I can't answer this question given these activations.
[epistemic status: 72] The user is asking about Python debugging.
[epistemic status: 15] The question is about cooking (but model is uncertain)"""
