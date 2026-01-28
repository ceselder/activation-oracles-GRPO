"""Handle epistemic status format for oracle outputs.

Format: [epistemic status: X.XX] <answer>

Examples:
    [epistemic status: 0.85] The text discusses a long-distance relationship...
    [epistemic status: 0.3] This appears to be about machine learning...
    [epistemic status: 0.1] I cannot determine the topic from these activations.
"""

import re
from dataclasses import dataclass


@dataclass
class OracleOutput:
    """Parsed oracle output with confidence and answer."""

    confidence: float
    answer: str
    raw: str
    parse_success: bool


# Regex to parse epistemic status
EPISTEMIC_PATTERN = re.compile(
    r"^\s*\[epistemic status:\s*([01]?\.?\d*)\s*\]\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


def parse_oracle_output(raw_output: str, default_confidence: float = 0.5) -> OracleOutput:
    """Parse oracle output into confidence and answer.

    If parsing fails, returns default_confidence (0.5) which penalizes
    bad formatting naturally through the Brier score.

    Args:
        raw_output: Raw string from oracle generation
        default_confidence: Confidence to use if parsing fails

    Returns:
        OracleOutput with parsed or default values
    """
    raw_output = raw_output.strip()

    match = EPISTEMIC_PATTERN.match(raw_output)
    if match:
        try:
            confidence = float(match.group(1))
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
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


def format_epistemic_output(confidence: float, answer: str) -> str:
    """Format answer with epistemic status prefix.

    Args:
        confidence: Confidence level (0-1)
        answer: The answer text

    Returns:
        Formatted string with epistemic status prefix
    """
    return f"[epistemic status: {confidence:.2f}] {answer}"


# System prompt to teach the oracle the format
ORACLE_SYSTEM_PROMPT = """You are an Activation Oracle. You receive hidden activations from a language model and answer questions about what the model was processing.

You MUST always respond in this exact format:
[epistemic status: X.XX] Your answer here

Where X.XX is a number between 0.00 and 1.00 representing your confidence.

Guidelines for epistemic status:
- 0.90-1.00: Very confident, clear signal in activations
- 0.70-0.89: Confident, strong evidence
- 0.50-0.69: Moderate confidence, some uncertainty
- 0.30-0.49: Low confidence, weak or ambiguous signal
- 0.10-0.29: Very uncertain, mostly guessing
- 0.00-0.09: Cannot determine from activations

Be CALIBRATED: your confidence should match your actual accuracy. If you're often wrong when confident, lower your confidence. If you're often right when uncertain, raise it.

Examples:
[epistemic status: 0.85] The text discusses a long-distance relationship between two people navigating career uncertainty.
[epistemic status: 0.3] This appears to be about machine learning, but the activations are ambiguous about the specific subfield.
[epistemic status: 0.1] I cannot determine the topic from these activations."""
