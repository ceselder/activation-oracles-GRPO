# WildChat Oracle Questions Dataset

Training data for activation oracles: user prompts paired with semantic probe questions.

## Design Principle

Questions probe **SEMANTIC CONTENT** encoded in neural activations - what concepts, emotions, and intentions are being processed. ~50% have negative answers to test calibration.

## Question Types

- **Topic/domain**: "What is the main topic?", "Is this about cooking?"
- **Emotion/sentiment**: "Is the user frustrated?", "What mood is conveyed?"
- **Intent/goal**: "What does the user want?", "Is this asking for help?"
- **Inference**: "Does the user seem experienced?", "Is this from a student?"
- **Subtle distinctions**: "Confused or curious?", "Complaint or question?"
- **Negative probes**: Wrong topics/emotions to get "No" answers

## Stats
- Examples: 100
- Total questions: 904
- Binary questions: 757 (83.7%)
- Generation cost: $0.0427
- Model: google/gemini-3-flash-preview

## Format
```json
{
  "wildchat_question": "user message...",
  "language": "english",
  "oracle_questions": ["What is the main topic?", "Is the user frustrated?", ...]
}
```

## Usage
```python
from datasets import load_dataset
ds = load_dataset("ceselder/wildchat-oracle-questions")
```
