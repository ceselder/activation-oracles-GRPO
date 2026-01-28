# WildChat Oracle Questions Dataset

Training data for activation oracles: user prompts paired with diverse probe questions.

## Key Features
- **~50% negative questions**: Questions designed to have "No" answers (probing wrong topics/emotions)
- **Mixed question types**: Binary (yes/no) and open-ended questions
- **Diverse probes**: Topic, sentiment, intent, author traits, tone, content type

## Stats
- Examples: 1000
- Total questions: 7997
- Binary questions: 4215 (52.7%)
- Open-ended: 3782 (47.3%)
- Generation cost: $0.0563

## Format
```json
{
  "wildchat_question": "user message...",
  "language": "english",
  "oracle_questions": ["What is the model thinking about?", "Is this about cooking?", ...]
}
```

## Usage
```python
from datasets import load_dataset
ds = load_dataset("ceselder/wildchat-oracle-questions")
```
