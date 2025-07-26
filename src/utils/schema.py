class CaptionSchema:
    CAPTION = "caption"
    IMAGE = "image"


class PickleSchema:
    CAPTION_MODEL_STATE_DICT = "caption_model_state_dict"
    GPT2_CONFIG = "gpt2_config"
    CLIP_CONFIG = "clip_config"
    TRAINING_CONFIG = "training_config"
    TRAIN_LOSS = "train_loss"
    VALIDATION_LOSS = "validation_loss"
    VALIDATION_BERT_PRECISION = "validation_bert_precision"
    VALIDATION_BERT_RECALL = "validation_bert_recall"
    VALIDATION_BERT_F1_SCORE = "validation_bert_f1_score"
