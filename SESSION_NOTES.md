# ReST Activation Oracle Training Session Notes

## Current State
- **Model**: Gemma 3 27B with AO checkpoint (`adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it`)
- **Dataset**: WildChat (English only)
- **Questions**: 20 per prompt, includes "What is the model thinking about?"
- **Adaptive batching**: Implemented (auto-reduces on OOM)

## Key Findings
1. **Question quality improved** with simpler templates that explicitly require question marks
2. **Large norm warnings** during generation are expected (activation steering)
3. **Speed issue**: 27B is slow for question generation (~8s/prompt on H100)

## Next Steps: Outsource to Gemini 2.5 Flash
Outsource question generation and judging to Gemini Flash via OpenRouter:
- **Cost**: ~$0.08 per round, ~$2-5 for full training
- **Benefit**: Much faster, frees GPU for training only

### Implementation Plan
1. Create `rest_ao/external_llm.py` with:
   - `generate_questions_external(prompts, api_key)` using Gemini Flash
   - `judge_responses_external(samples, api_key)` using Gemini Flash
2. Update `rest_trainer.py` to use external LLM for questions/judging
3. Keep local model for activation extraction + response generation only

## Files Modified
- `rest_ao/config.py` - Gemma 3 27B, batch sizes, question templates
- `rest_ao/data_pipeline.py` - English-only WildChat filter
- `rest_ao/rest_trainer.py` - Gemma 3 layer access fix, debug prints
- `rest_ao/question_generation.py` - Adaptive batching
- `train.py` - Default model/checkpoint

## SSH Access
```bash
ssh -p 25878 root@192.222.52.140 -L 8080:localhost:8080
```

## API Keys (in .env)
- HF_TOKEN, OPENROUTER_API_KEY configured
- WANDB_API_KEY empty (was running with WANDB_MODE=disabled)
