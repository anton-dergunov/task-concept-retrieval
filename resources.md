# Resources

Background reading collected while scoping this project. Most of it predates the current
generation of embedding models and is kept for the framing and the datasets rather than the
methods.

## Icon vocabulary

- Material Symbols — the icon set the agenda will actually display, and therefore the label
  space the gold set should be annotated against.
  https://fonts.google.com/icons

## Emoji semantic search (the earlier framing of this problem)

- "How to Build a Semantic Search Engine for Emojis"
  https://medium.com/data-science/how-to-build-a-semantic-search-engine-for-emojis-ef4c75e3f7be
- "Semantic search for emojis in 50 languages using AI"
  https://towardsdatascience.com/semantic-search-for-emojis-in-50-languages-using-ai-f85a36a86f21/
- "Emoji search: creating, deploying and evaluating a machine learning service"
  https://medium.com/nerd-for-tech/emoji-search-creating-deploying-and-evaluating-a-machine-learning-service-f97c29976d6d
- Reddit discussion, "What is a good emoji-aware pretrained language model?"
  https://www.reddit.com/r/MachineLearning/comments/u99fgu/d_what_is_a_good_emoji_aware_pretrained_language/

## Emoji-aware models and tooling

- spaCy emoji pipeline component
  https://github.com/explosion/spacymoji
- Multilingual BERT fine-tuned for emoji prediction
  https://huggingface.co/jirmauritz/bert-multilingual-emoji
- Emoji prediction with deep learning (code)
  https://github.com/Defcon27/Emoji-Prediction-using-Deep-Learning

## Papers

- "Using millions of emoji occurrences to learn any-domain representations for detecting
  sentiment, emotion and sarcasm" (DeepMoji, 2017) — the origin of emoji-as-supervision
  https://arxiv.org/abs/1708.00524
- Emoji representation survey (ScienceDirect)
  https://www.sciencedirect.com/science/article/pii/S2666449621000529
- Recent emoji-semantics work (2024)
  https://arxiv.org/html/2409.10760v1

## Retrieval method references

- MTEB leaderboard — for choosing which embedding models to include as arms
  https://huggingface.co/spaces/mteb/leaderboard
- sentence-transformers documentation, in particular hard-negative mining and cross-encoder
  re-ranking
  https://sbert.net/
