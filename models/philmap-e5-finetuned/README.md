---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:94
- loss:CosineSimilarityLoss
base_model: intfloat/multilingual-e5-base
widget:
- source_sentence: 'dharma | dharma (cosmic law/duty) | The cosmic law, moral order,
    and the Buddha''s teaching. | Tradition: Buddhism'
  sentences:
  - 'salvation | Deliverance from sin and its consequences through divine grace. |
    Tradition: Christianity'
  - 'predestination | predestination | Tradition: Calvinism'
  - '間柄 | aidagara (betweenness) | 人間存在の根本構造としての、人と人との間の関係性。個人は間柄を通じてのみ自己を実現する。 |
    The relational space between persons that constitutes their individuality — neither
    isolated individual nor undifferentiated collectivity. | 和辻は「人間」の語源（人の間）から、存在が本質的に関係的であることを論じた
    | 間柄は倫理の基盤であり、空間的・社会的な「あいだ」を含む | Tradition: Kyoto School'
- source_sentence: 'natural law | The rational creature''s participation in eternal
    law, directing human action toward the good. | Tradition: Scholasticism'
  sentences:
  - 'das Nichts | the Nothing | Das Nichts ist nicht das Gegenteil des Seienden, sondern
    gehört ursprünglich zum Wesen selbst. | The Nothing is not the opposite of beings
    but belongs originally to the essence of Being itself. | Tradition: Continental
    Phenomenology'
  - 'πνεῦμα | pneuma (breath/spirit) | The vital breath or creative fire pervading
    all things, giving them cohesion and life. | Tradition: Stoicism'
  - 'ṛta | cosmic order | The cosmic order governing nature and morality. | Tradition:
    Vedic'
- source_sentence: 'dharma | dharma (cosmic law/duty) | The cosmic law, moral order,
    and the Buddha''s teaching. | Tradition: Buddhism'
  sentences:
  - 'Schein | appearance/illusion | Der transzendentale Schein: die unvermeidliche
    Illusion der reinen Vernunft. | Transcendental illusion: the unavoidable illusion
    of pure reason that mistakes appearances for things-in-themselves. | Tradition:
    Kantian'
  - 'karuṇā | compassion | Compassion for all sentient beings. Active wish that others
    be free from suffering. | Tradition: Buddhism'
  - 'pratītyasamutpāda | dependent origination | All phenomena arise in dependence
    upon causes and conditions. Nothing exists independently. | Tradition: Buddhism'
- source_sentence: '場所 | basho (place/topos) | あらゆる存在者がそこにおいて成立する場所。意識の底にある主客未分の根源的な場。
    | The place in which all beings come to be. The ultimate basho is absolute nothingness.
    | Tradition: Kyoto School'
  sentences:
  - '理 | li (principle) | 天地万物的所以然之理。理在气先。 | The principle or pattern inherent in
    all things. Li precedes qi. | Tradition: Neo-Confucianism'
  - 'māyā | illusion/cosmic power | The cosmic power of illusion that makes the one
    Brahman appear as the manifold world. | Tradition: Advaita Vedanta'
  - '場所 | basho (place/topos) | あらゆる存在者がそこにおいて成立する場所。意識の底にある主客未分の根源的な場。 | The place
    in which all beings come to be. The ultimate basho is absolute nothingness. |
    Tradition: Kyoto School'
- source_sentence: '無為 | 無為 | Tradition: Daoism'
  sentences:
  - 'nirvāṇa | nirvana (extinction) | The cessation of suffering and the cycle of
    rebirth. Extinguishing of craving, aversion, and delusion. | Tradition: Buddhism'
  - 'pratītyasamutpāda | dependent origination | All phenomena arise in dependence
    upon causes and conditions. Nothing exists independently. | Tradition: Buddhism'
  - 'categorical imperative | categorical imperative | Tradition: Kantian'
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
      name: phil concept eval
      type: phil-concept-eval
    metrics:
    - type: pearson_cosine
      value: 0.8528596332935514
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.8451542547285167
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
    '無為 | 無為 | Tradition: Daoism',
    'categorical imperative | categorical imperative | Tradition: Kantian',
    'pratītyasamutpāda | dependent origination | All phenomena arise in dependence upon causes and conditions. Nothing exists independently. | Tradition: Buddhism',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2765, 0.3542],
#         [0.2765, 1.0000, 0.3805],
#         [0.3542, 0.3805, 1.0000]])
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

* Dataset: `phil-concept-eval`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.8529     |
| **spearman_cosine** | **0.8452** |

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

* Size: 94 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 94 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          | label                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | float                                                          |
  | details | <ul><li>min: 12 tokens</li><li>mean: 53.15 tokens</li><li>max: 130 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 46.56 tokens</li><li>max: 130 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.47</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                    | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>場所 \| basho (place/topos) \| あらゆる存在者がそこにおいて成立する場所。意識の底にある主客未分の根源的な場。 \| The place in which all beings come to be. The ultimate basho is absolute nothingness. \| Tradition: Kyoto School</code>                                   | <code>χώρα \| chōra (receptacle) \| The receptacle or space in which Forms are instantiated. \| Tradition: Platonism</code>                                                   | <code>1.0</code> |
  | <code>natural law \| The rational creature's participation in eternal law, directing human action toward the good. \| Tradition: Scholasticism</code>                                                                                   | <code>nirvāṇa \| nirvana (extinction) \| The cessation of suffering and the cycle of rebirth. Extinguishing of craving, aversion, and delusion. \| Tradition: Buddhism</code> | <code>0.2</code> |
  | <code>絶対無 \| absolute nothingness \| あらゆる有を包み、自らは対象化されない究極の場所。主客未分の根源的な場。 \| The ultimate place that envelops all being without itself being determined. The ground prior to subject-object division. \| Tradition: Kyoto School</code> | <code>فناء \| fanāʾ (annihilation) \| Annihilation of the ego-self in the divine. Cessation of self-will. \| Tradition: Sufi Philosophy</code>                                | <code>0.2</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 10
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
| Epoch | Step | phil-concept-eval_spearman_cosine |
|:-----:|:----:|:---------------------------------:|
| 1.0   | 12   | 0.5071                            |
| 2.0   | 24   | 0.8452                            |


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