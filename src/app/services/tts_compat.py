"""
Compatibility patch for TTS to work with newer transformers versions (>=4.46.2).
This fixes the BeamSearchScorer import issue by making it available at the root level.

TTS 0.22.0 tries to import BeamSearchScorer from transformers root, but in newer
versions it's only available in transformers.generation.beam_search.
"""
import transformers

# Check if BeamSearchScorer is available in the root module
if not hasattr(transformers, 'BeamSearchScorer'):
    try:
        # In transformers >=4.46.2, BeamSearchScorer is in generation.beam_search
        from transformers.generation.beam_search import BeamSearchScorer
        # Make it available at the root level for TTS compatibility
        transformers.BeamSearchScorer = BeamSearchScorer
        # Also ensure it's accessible via generation module
        if not hasattr(transformers.generation, 'BeamSearchScorer'):
            transformers.generation.BeamSearchScorer = BeamSearchScorer
    except ImportError:
        try:
            # Try alternative import path (older versions)
            from transformers.generation import BeamSearchScorer
            transformers.BeamSearchScorer = BeamSearchScorer
        except ImportError:
            # If still not found, this is a problem
            import warnings
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "BeamSearchScorer not found in transformers. TTS may not work. "
                f"Transformers version: {transformers.__version__}"
            )

