---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:138
- loss:CosineSimilarityLoss
base_model: intfloat/multilingual-e5-base
widget:
- source_sentence: 'आत्मन् Atman | True self, identical with Brahman in Advaita |
    Tradition: Vedanta'
  sentences:
  - '理 li | Principle inherent in all things, li precedes qi | Zhu Xi | Tradition:
    Neo-Confucianism'
  - '身心脱落 shinjin datsuraku | Dogen: dropping off body and mind in seated meditation
    | Tradition: Chan/Zen'
  - '仁 ren | Benevolence humaneness, the consummate person exemplifies ren | Confucius
    | Tradition: Confucianism'
- source_sentence: 'आत्मन् Atman | True self, identical with Brahman in Advaita |
    Tradition: Vedanta'
  sentences:
  - 'धर्मकीर्ति Dharmakirti epistemology | Perception and inference as the two valid
    means of knowledge | Buddhist logic | Tradition: Buddhism'
  - 'अनेकान्तवाद anekantavada | Reality is complex, perceived from multiple perspectives,
    none complete alone | Tradition: Jainism'
  - 'pratītyasamutpāda dependent origination | All phenomena arise dependently | Nagarjuna
    | Tradition: Buddhism'
- source_sentence: 'λόγος | logos (reason/order) | The rational principle governing
    the cosmos. The underlying order of all things. | Tradition: Presocratic'
  sentences:
  - '唯識 vijnaptimatrata | All phenomena are consciousness-only, no external objects
    | Vasubandhu | Tradition: Yogacara'
  - categorical imperative | Kantian
  - 'ṛta | cosmic order | The cosmic order or truth that governs both the natural
    and moral realms. | Tradition: Vedic'
- source_sentence: 'अनेकान्तवाद anekantavada | Reality is complex, perceived from
    multiple perspectives, none complete alone | Tradition: Jainism'
  sentences:
  - '無 wu mu | Nothingness, non-being, Zen mu | Joshu''s Mu | Tradition: Chan/Zen
    Buddhism'
  - '兼爱 jian ai | Universal impartial love for all without distinction | Mozi | Tradition:
    Mohism'
  - 'karuṇā compassion | Compassion for all sentient beings, wish that others be free
    from suffering | Mahayana | Tradition: Buddhism'
- source_sentence: '理 li | Principle inherent in all things, li precedes qi | Zhu
    Xi | Tradition: Neo-Confucianism'
  sentences:
  - '理 li | Principle inherent in all things, li precedes qi | Zhu Xi | Tradition:
    Neo-Confucianism'
  - 'आत्मन् Atman | True self, identical with Brahman in Advaita | Tradition: Vedanta'
  - 'mokṣa liberation | Liberation from the cycle of birth and death | Tradition:
    Vedanta'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on intfloat/multilingual-e5-base
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: phil expanded eval
      type: phil-expanded-eval
    metrics:
    - type: pearson_cosine
      value: 0.995893541827785
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.9759000729485332
      name: Spearman Cosine
---

# SentenceTransformer based on intfloat/multilingual-e5-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) <!-- at revision 835193815a3936a24a0ee7dc9e3d48c1fbb19c55 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '理 li | Principle inherent in all things, li precedes qi | Zhu Xi | Tradition: Neo-Confucianism',
    'आत्मन् Atman | True self, identical with Brahman in Advaita | Tradition: Vedanta',
    'mokṣa liberation | Liberation from the cycle of birth and death | Tradition: Vedanta',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.1611, 0.1698],
#         [0.1611, 1.0000, 0.1900],
#         [0.1698, 0.1900, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `phil-expanded-eval`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.9959     |
| **spearman_cosine** | **0.9759** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 138 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 138 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | float                                                          |
  | details | <ul><li>min: 7 tokens</li><li>mean: 35.24 tokens</li><li>max: 100 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 33.25 tokens</li><li>max: 55 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.43</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                | sentence_1                                                                                                                  | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>śūnyatā 空 emptiness \| Emptiness of inherent existence, dependent origination \| Nagarjuna \| Tradition: Buddhism Madhyamaka</code> | <code>仁 ren \| Benevolence humaneness, the consummate person exemplifies ren \| Confucius \| Tradition: Confucianism</code> | <code>0.2</code> |
  | <code>Dasein \| Being that understands Being, being-in-the-world \| Heidegger Sein und Zeit \| Tradition: Phenomenology</code>            | <code>理 li \| Principle inherent in all things, li precedes qi \| Zhu Xi \| Tradition: Neo-Confucianism</code>              | <code>0.2</code> |
  | <code>mokṣa liberation \| Liberation from the cycle of birth and death \| Tradition: Vedanta</code>                                       | <code>आत्मन् Atman \| True self, identical with Brahman in Advaita \| Tradition: Vedanta</code>                             | <code>0.2</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 15
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 15
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | phil-expanded-eval_spearman_cosine |
|:-----:|:----:|:----------------------------------:|
| 1.0   | 18   | 0.9515                             |
| 2.0   | 36   | 0.8539                             |
| 3.0   | 54   | 0.9515                             |
| 4.0   | 72   | 0.9515                             |
| 5.0   | 90   | 0.9515                             |
| 6.0   | 108  | 0.9515                             |
| 7.0   | 126  | 0.9759                             |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.3.0
- Transformers: 5.3.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.8.3
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->