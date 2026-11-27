# `lgbm_staged_rfe` — Staged LightGBM RFE for 1.3M × 3,500 Datasets

A memory-safe, three-stage coarse-to-fine Recursive Feature Elimination (RFE) pipeline built for **LightGBM** on **large imbalanced binary classification datasets** (~1.3M rows × 3,500 columns, ~8% positives, 5 expanding time folds) running on high-RAM multi-core CPU instances (e.g., SageMaker 48-core / 256–512 GB RAM).

---

## Why Standard `sklearn.RFECV` Fails at This Scale

1. **Compute:** `RFECV(step=1)` with 5 folds requires ~17,500 full model fits.
2. **Memory:** 1.3M × 3,500 in `float64` is ~36.4 GB raw; `RFECV(..., n_jobs=-1)` forks worker processes and pandas copies that quickly blow past RAM or thrash cache.
3. **Methodological Bias:** Standard `RFECV` ranks by *in-sample training* `feature_importances_`, which favors high-cardinality or temporally unstable features.

---

## How `lgbm_staged_rfe` Solves It

| Stage | Feature Range | Rows Used | Step Size | Ranking Signal |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (optional)** | 3,500 | 100% (1.3M) | 1 evaluation | Apples-to-apples CV AUC reference line |
| **Stage 1: Coarse** | 3,500 → ~300 | 300K/fold (100% positives + time-stratified weighted negatives) | 25% geometric drop (or Null Importance) | Validation TreeSHAP stability (`max_bin=127`, `colsample=0.25`) |
| **Optional Collinearity Filter** | ~300 → ~200–250 | 50K sample | Hierarchical Spearman cluster dedup | Keeps top importance feature per tight cluster |
| **Stage 2: Medium** | ~300 → ~70 | 100% (1.3M) | 15% geometric drop | Validation TreeSHAP stability (`mean - 0.25 * std`) |
| **Stage 3: Fine** | ~70 → ~10 | 100% (1.3M) | **1-feature exact step** | Validation TreeSHAP stability across all 5 expanding folds |

### Memory & Speed Optimizations
- **`prepare_matrix()`** pre-allocates a single `float32` Fortran-contiguous NumPy array column-by-column (~18.2 GB for 1.3M × 3,500, avoiding any 36 GB `float64` copy).
- **Sequential folds + `n_jobs=48` inside LightGBM:** Zero duplicate matrix copies across CV worker processes.
- **Validation TreeSHAP (`pred_contrib=True`)** on up to `shap_max_val_rows=50_000` stratified validation rows per fold, while **ROC AUC is always scored on 100% of validation rows**.
- **Checkpointing & Resume:** Saves `history.csv`, `feature_sets.json`, and `state.json` after every step so long SageMaker runs can resume safely.

---

## Quickstart

```python
from lgbm_staged_rfe import (
    StagedLGBMRFE,
    StagedRFEConfig,
    make_expanding_time_splits,
    plot_fold_stability,
    plot_rfe_curve,
    plot_top_features,
    prepare_matrix,
)

# 1. Split off OOT holdout and build 5 expanding time folds
cv_splits, _ = make_expanding_time_splits(dev_df["vintage_month"], n_splits=5)
prepared_dev = prepare_matrix(dev_df, feature_cols=feature_cols)

# 2. Configure and run
config = StagedRFEConfig(
    stage1_target=300,
    stage2_target=70,
    min_features=15,
    n_jobs=48,
    checkpoint_dir="./rfe_checkpoints",
)
selector = StagedLGBMRFE(config).fit(
    prepared_dev,
    dev_df["target"].to_numpy(),
    cv_splits=cv_splits,
    time_values=dev_df["vintage_month"].to_numpy(),
)

# 3. Visually choose feature count
plot_rfe_curve(
    selector.results_,
    baseline_auc=selector.baseline_result_["mean_auc"],
    target_budget=50,
    zoom_max_features=120,
    save_path="rfe_curve.png",
)
plot_fold_stability(selector.results_, zoom_max_features=120, target_budget=50, save_path="fold_stability.png")

# 4. Extract exact feature list for your chosen count
chosen_features = selector.get_features(50)
plot_top_features(selector.get_feature_importances(50), top_n=50, save_path="top_50.png")
```

Run the fast local synthetic demo and test suite:

```bash
python examples/sagemaker_usage.py --synthetic --output-dir output_demo
pytest -v
```









Partindo dessa última versão do artefato HTML gostaria de implementar as seguintes modificações:
  - Página 1:
    - Trazer a visualização da árvore para a esquerda e as demais métricas para a direita;
    - O título do container da árvore deve ser "Segmentação do público de desenvolvimento"
    - Queria propor uma mudança para a estrutura da árvore no geral (ainda parecido com o que temos atualmente): a ideia é transformar o que temos atualmente em um "icicle chart" horizontal, onde teremos na extrema esquerda um retângulo grande de "Público" com a volumetria total indicada; a segmentação de produto vem logo em seguida e os retângulos representando os 2 produtos (CONSIG_EP, CREDIARIO_CP) aparecem ao lado do inicial e um acima do outro suas alturas somadas correspondem a altura do anterior sendo que as alturas individuais representam a proporção de público que se distibui entre cada um deles; a segmentação de bancos, com seus 2 valores, vai ser repetida para cada um dos produtos... a altura somada dos 4 retÂngulos empilhados equivale a soma das alturas dos retângulos dos 2 produtos (e a altura total do de público) enquanto os retângulos individuais representarão a proporção de público para o produto a sua esquerda E a categoria de bancos representada; e assim por diante. A interação de seleção pode ser mantida muito igual ao já existente, com o destaque da cor nos retângulos selecionados, 1 selecionado por nível; ao chegarmos no nível das folhas (quebra por faixas de idade) aí devemos poder ver somente a quebra de idades para as segmentações já selecionadas (ou seja, uma abertura somente de 5/6 folhas por vez);
    - Acima da árvore deve ser disponibilizado um seletor com as opções de Total, Treino e Teste que ao ser selecionada deve atualizar as volumetrias de população de cada segmentação para refletir a visão geral, os números somente para o conjunto de treino/teste (lembrando que o conjunto de treino corresponde aos dados de 202505 a 202509 e teste de 202510 a 202512 limites inclusivos);
    - Produzir uma pequena animação que, mediante um botão PLAY, execute uma passagem simulada da base completa pelo critério de preenchimento, por exemplo; uma legenda apresentada acima da árvore vai ser atualizada durante o processo descrevendo as etapas do que ocorre; ao dar play, a legenda inicial poderia ser "Seleção de variáveis com segmentação hierárquica" e começaria mostrando somente as 5/6 folhas mais superiores da árvore selecionadas e abertas; avança o quadro para "Avaliando o preenchimento de uma única feature", ao lado direito das folhas que se encontram abertas são indicados valores (que podem ser fictícios inclusive) de porcentagem para indicar que naquela segmentação de público específica estamos verificando preenchimento de x% de linhas, uma indicação logo acima das folhas em fonte menor que a da legenda traz a informação de que os valores representam a % de preenchimento da coluna para cada segmento; enquanto a legenda ainda é a mesma, nas folhas cujo valor de % é < 5% a gente marca de vermelho e as demais de verde; a legenda muda para "A métrica é avaliada para todas as segmentações disponíveis" enquanto isso a animação vai recolhendo as folhas anteriores, abrindo a distribuição das próximas, mostrando o valor de %, marcando com a cor apropriada, recolhendo o grupo de folhas e assim por diante enquanto desce ao longo das diferentes segmentações; a animação abre uma segmentação onde existe uma folha N_A para idade e para por um instante, a legenda atualiza para "A métrica só é avaliada em segmentações que possuem um mínimo de 500 observações totais e de 100 eventos de default"; a folha N_A de baixa volumetria recebe a cor cinza, seu grupo é recolhido e a descida ao longo dos níveis da árvore continua; ao alcançar o extremo inferior e marcadas todas as folhas de verde/vermelho/cinza a árvore fecha todas as folhas (que mantém um pequeno "risquinho" da cor atribuída visível quando a segmentação é recolhida, só pra facilitar o entendimento) e nisso a árvore com todas as folhas fechadas deslocase um pouco para a esquerda e fica mais transparente enquanto três retangulos com as devidas cores assumem sua extremidade direita (com um certo espaçamento relativo a arvore) mantendo a soma das suas alturas igual a altura total da árvore, como se agora representasssem a soma de todas as folhas segmentadas por cinza/vermelho/verde mostrando o número total de folhas de cada cor em seu respectivo interior, enquanto a legenda atualiza para "O resultado final leva em conta todas as segmentações avaliadas"; os retangulos verde e vermelho estarão acima e o cinza mais abaixo, uma linha mais grossa será traçada verticalmente ao lado direito dos retângulos cobrindo somente a altura dos 2 retângulos superiores, e a sua direita começa a aparecer o texto "A cada etapa, são mantidas as variáveis que satisfazem o critério para PELO MENOS UMA segmentação válida", e enquanto o texto vai aparecendo os retângulos verde e vermelho (já sem a indicação de número de folhas em seu interior somente a cor mesmo) começam a variar de tamanho enquanto o espaço horizontal entre eles começa a oscilar um pouco maus pra cima depois um pouco mais pra baixo (sempre mantendo estáticos o limite inferior do retângulo inferior e o limite superior do retângulo superior); depois de um tempinho o retângulo vermelho toma o espaço todo e o verde some enquanto o texto do lado direito vai atualizando para "Eliminam-se somente as váriávies que não satisfazem o critério para NENHUMA das segmentações válidas"; finalizando aqui, uma transição suave que traz a árvore de volta para sua posição inicial;
    - Nas tabelas onde temos as % de preenchimento por books nas segmentações de idade e produto ao início e ao final do processo gostaria que fosse acresentado um seletor para filtrar segmentação por produto, aí só são exibidos os valores quebrados para os 2 produtos; ao selecionar faixa de idade aí ambas tabelas atualizam para mostrar os valores quebrados nas diferentes faixas
