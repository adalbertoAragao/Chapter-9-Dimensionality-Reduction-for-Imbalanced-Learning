![logo_unb.png](ee6922b9-03cf-4da8-aec0-937bac745554.png)

# <center>Departamento de Ciência da Computação</center>

# <center>PPCA - Programa de Pós-Graduação em Computação Aplicada</center>

# <center>Exercício 1</center>

# <center>Chapter 9: Dimensionality Reduction for Imbalanced Learning</center>

# <center>Book: Learning from Imbalanced Data Sets</center>

# <center>Fernández et al., 2018</center>

## <center>15 de agosto de 2026</center>

## <center>Prof. Marcelo Ladeira</center>
=====================================================
=====================================================
 - #### <font color="blue">Aluno: Adalberto Araujo Aragão</font>
 - #### <font color="blue">Disciplina: Mineração de Dados</font>
 - #### <font color="blue">Turma: turma 1, 2º Semestre de 2026</font>
 - #### <font color="blue">Data de entrega: 5 de setembro de 2026</font>
=====================================================
=====================================================

# Redução de Dimensionalidade para Aprendizado Desbalanceado
### Capítulo 9 — *Learning from Imbalanced Data Sets*
Fernández, García, Galar, Prati, Krawczyk & Herrera (2018)

Este notebook acompanha a apresentação em sala de aula e reproduz, em Python, as ideias centrais do capítulo:

1. Motivação: alta dimensionalidade + desbalanceamento
2. Seleção de atributos (Feature Selection) — ReliefF e comparação com métodos clássicos
3. Efeito do SMOTE em alta dimensão (achado citado no capítulo)
4. Extração de atributos linear — PCA Assimétrica (APCA)
5. Extração de atributos não linear — Autoencoders (framework DAF)
6. Discretização sensível ao desbalanceamento — critério CAIM / ur-CAIM
7. Conclusões

> **Observação:** os dados usados são **sintéticos** (gerados com `make_classification`), o que garante reprodutibilidade e tempo de execução curto para a apresentação. As conclusões qualitativas seguem a discussão do capítulo, mas os números exatos dependem do dataset gerado.


```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from imblearn.over_sampling import SMOTE

np.random.seed(42)
plt.rcParams["figure.figsize"] = (6, 4)
print("Bibliotecas carregadas com sucesso.")
```

    Bibliotecas carregadas com sucesso.
    

## 1. Motivação (Seção 9.1)

O capítulo destaca que dados de **alta dimensionalidade** (muitas variáveis, poucas amostras — comum em dados de microarray/biomédicos) tornam o problema de desbalanceamento ainda mais difícil:

- Classificadores padrão tendem a favorecer a classe majoritária, e esse viés **aumenta** com a dimensionalidade.
- O **SMOTE** perde eficácia (ou até piora o desempenho) em espaços de alta dimensão, a menos que seja feita uma seleção de atributos antes.
- O **k-NN** sofre do fenômeno de *hubness* (poucas instâncias dominam as vizinhanças), o que piora ainda mais em altas dimensões.

Vamos gerar um dataset sintético que imita esse cenário: **40 atributos**, apenas **6 informativos**, e desbalanceamento de aproximadamente **9:1**.


```python
X, y = make_classification(
    n_samples=1000, n_features=40, n_informative=6, n_redundant=10,
    n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0.02,
    class_sep=0.7, shuffle=False, random_state=42
)

print("Formato de X:", X.shape)
print("Distribuição de classes:", Counter(y))
print(f"Razão de desbalanceamento: 1:{Counter(y)[0]/Counter(y)[1]:.1f}")

# shuffle=False garante que as 6 primeiras colunas sejam de fato os atributos informativos,
# o que nos permite depois comparar com uma seleção "perfeita" de referência.
Xs = MinMaxScaler().fit_transform(X)  # normalização em [0,1], usada nas próximas seções
```

    Formato de X: (1000, 40)
    Distribuição de classes: Counter({np.int64(0): 891, np.int64(1): 109})
    Razão de desbalanceamento: 1:8.2
    

## 2. Seleção de Atributos: ReliefF (Seção 9.2.1, Algoritmo 1)

O **ReliefF** avalia cada atributo com base em quão bem ele separa instâncias de classes diferentes ("misses") mantendo próximas instâncias da mesma classe ("hits"). Reproduzimos o pseudocódigo do Algoritmo 1 diretamente em Python.


```python
def diff(f, a, b, ranges):
    """Diferença normalizada entre duas instâncias no atributo f (função diff() do Algoritmo 1)."""
    r = ranges[f]
    return 0.0 if r == 0 else abs(a[f] - b[f]) / r


def relieff(X, y, k=8, n_iter=200, seed=0):
    """Implementação direta do Algoritmo 1 (ReliefF)."""
    rng = np.random.default_rng(seed)
    m, n = X.shape
    ranges = X.max(axis=0) - X.min(axis=0)
    probs = {c: cnt / m for c, cnt in Counter(y).items()}
    W = np.zeros(n)
    idx_all = np.arange(m)

    for _ in range(n_iter):
        i = rng.integers(0, m)
        xi, ci = X[i], y[i]

        # k vizinhos mais próximos da MESMA classe (hits)
        same = idx_all[(y == ci) & (idx_all != i)]
        d_same = np.linalg.norm(X[same] - xi, axis=1)
        hits = same[np.argsort(d_same)[:k]]

        # k vizinhos mais próximos de CADA classe diferente (misses)
        miss_by_class = {}
        for C in probs:
            if C == ci:
                continue
            other = idx_all[y == C]
            d_other = np.linalg.norm(X[other] - xi, axis=1)
            miss_by_class[C] = other[np.argsort(d_other)[:k]]

        for f in range(n):
            hit_term = sum(diff(f, xi, X[h], ranges) for h in hits) / (m * k)
            miss_term = 0.0
            for C, midx in miss_by_class.items():
                w_c = probs[C] / (1 - probs[ci])
                miss_term += w_c * sum(diff(f, xi, X[mi], ranges) for mi in midx)
            miss_term /= (m * k)
            W[f] += miss_term - hit_term

    return W


W_relief = relieff(Xs, y, k=8, n_iter=300, seed=1)
ranking_relief = np.argsort(-W_relief)
print("Top 10 atributos segundo ReliefF:", ranking_relief[:10].tolist())
print("Pesos correspondentes:          ", np.round(W_relief[ranking_relief[:10]], 4))
```

    Top 10 atributos segundo ReliefF: [5, 2, 6, 10, 1, 3, 8, 4, 9, 7]
    Pesos correspondentes:           [0.0178 0.0152 0.0149 0.0145 0.0144 0.014  0.0136 0.0135 0.0124 0.0114]
    

### Comparação com um método clássico (Mutual Information)

O capítulo observa (Seção 9.2.2.2) que métricas contínuas como **PCC**, **FAST** e **S2N** costumam superar abordagens que discretizam os atributos. Aqui comparamos o ranking do ReliefF com a **informação mútua**, uma métrica clássica e amplamente usada como filtro.


```python
mi = mutual_info_classif(Xs, y, random_state=0)
ranking_mi = np.argsort(-mi)
print("Top 10 atributos segundo Mutual Information:", ranking_mi[:10].tolist())

# Quantos atributos entre os 10 mais bem ranqueados coincidem nos dois métodos?
overlap = len(set(ranking_relief[:10]) & set(ranking_mi[:10]))
print(f"Sobreposição entre os Top-10 do ReliefF e do MI: {overlap}/10 atributos")
```

    Top 10 atributos segundo Mutual Information: [8, 7, 3, 6, 9, 2, 4, 5, 12, 10]
    Sobreposição entre os Top-10 do ReliefF e do MI: 9/10 atributos
    

## 3. Efeito do SMOTE em Alta Dimensão (Seção 9.1)

O capítulo cita o estudo de Blagus & Lusa: o SMOTE é eficaz em baixa dimensão, mas **perde efeito (ou piora) em alta dimensão**, a menos que seja feita uma seleção de atributos antes.

Vamos testar isso diretamente: comparamos o F1-score de um classificador k-NN **com e sem SMOTE**, primeiro usando todos os 40 atributos, depois usando apenas os 6 atributos originalmente informativos (simulando uma seleção de atributos perfeita).


```python
def avaliar_smote(X, y, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1_sem, f1_com = [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
        f1_sem.append(f1_score(yte, knn.predict(Xte)))

        Xr, yr = SMOTE(random_state=0).fit_resample(Xtr, ytr)
        knn2 = KNeighborsClassifier(n_neighbors=5).fit(Xr, yr)
        f1_com.append(f1_score(yte, knn2.predict(Xte)))

    print(f"{label:35s} | F1 sem SMOTE = {np.mean(f1_sem):.3f}  |  F1 com SMOTE = {np.mean(f1_com):.3f}")


avaliar_smote(Xs, y, "Alta dimensão (40 atributos)")
avaliar_smote(Xs[:, ranking_relief[:6]], y, "Após FS com ReliefF (6 atributos)")
```

    Alta dimensão (40 atributos)        | F1 sem SMOTE = 0.661  |  F1 com SMOTE = 0.454
    Após FS com ReliefF (6 atributos)   | F1 sem SMOTE = 0.809  |  F1 com SMOTE = 0.727
    

**Leitura do resultado:** com k-NN, o SMOTE tende a **piorar** o F1 nesse cenário sintético (efeito documentado na literatura: pontos sintéticos podem cair perto da fronteira de decisão e confundir vizinhos). O ponto central do capítulo, porém, se confirma: **o impacto negativo do SMOTE é muito mais acentuado em alta dimensão** do que depois que a dimensionalidade é reduzida pela seleção de atributos — ou seja, a interação entre SMOTE e dimensionalidade é real e relevante, reforçando a recomendação do capítulo de combinar FS com técnicas de balanceamento em vez de aplicar o SMOTE "cru" em dados de alta dimensão.

## 4. Extração Linear de Atributos: PCA Assimétrica — APCA (Seção 9.4.1)

Diferente do PCA tradicional (que pondera as classes pela sua frequência), a **APCA** pondera a matriz de covariância de forma **inversamente proporcional** ao tamanho da classe (Eq. 9.7), dando mais peso à classe menos representada — e portanto menos confiável estatisticamente:

$$\Sigma_\alpha = \alpha_0 \Sigma_0 + \alpha_c \Sigma_c + \Sigma_m, \qquad \alpha_0=\frac{q_c}{q},\ \alpha_c=\frac{q_0}{q}$$


```python
def apca_covariance(X, y):
    """Constrói a matriz de covariância assimétrica Sigma_alpha (Eqs. 9.6-9.7)."""
    X0, Xc = X[y == 0], X[y == 1]           # 0 = majoritária, c = minoritária
    q0, qc, q = len(X0), len(Xc), len(X)
    M0, Mc, M = X0.mean(0), Xc.mean(0), X.mean(0)

    Sigma0 = np.cov(X0, rowvar=False)
    Sigmac = np.cov(Xc, rowvar=False)
    Sigma_m = (q0 * np.outer(M0 - M, M0 - M) + qc * np.outer(Mc - M, Mc - M)) / q

    alpha0, alphac = qc / q, q0 / q          # pesos INVERSAMENTE proporcionais ao tamanho (Eq. 9.7)
    return alpha0 * Sigma0 + alphac * Sigmac + Sigma_m


Sigma_alpha = apca_covariance(Xs, y)
eigval, eigvec = np.linalg.eigh(Sigma_alpha)
order = np.argsort(-eigval)

m = 6
Phi_m = eigvec[:, order[:m]]
X_apca = Xs @ Phi_m
X_pca = PCA(n_components=m, random_state=0).fit_transform(Xs)

print("APCA:", X_apca.shape, " | PCA tradicional:", X_pca.shape)
```

    APCA: (1000, 6)  | PCA tradicional: (1000, 6)
    


```python
def avaliar_classificacao(Xfeat, y, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1s = []
    for tr, te in skf.split(Xfeat, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xfeat[tr], y[tr])
        f1s.append(f1_score(y[te], clf.predict(Xfeat[te])))
    print(f"{label:35s} | F1 = {np.mean(f1s):.3f}")


avaliar_classificacao(Xs, y, "Atributos originais (40)")
avaliar_classificacao(X_pca, y, "PCA tradicional (6 comp.)")
avaliar_classificacao(X_apca, y, "APCA (6 comp., Eq. 9.7)")
```

    Atributos originais (40)            | F1 = 0.456
    

    PCA tradicional (6 comp.)           | F1 = 0.441
    APCA (6 comp., Eq. 9.7)             | F1 = 0.501
    

**Ideia central:** ao ponderar mais a classe minoritária (menos amostras → covariância menos confiável), a APCA tenta preservar direções discriminativas que o PCA tradicional poderia descartar por variância aparente baixa.

## 5. Extração Não Linear: Autoencoders — Framework DAF (Seção 9.5)

O **DAF (Dual Autoencoding Features)** combina dois autoencoders empilhados com funções de ativação diferentes — **sigmoid** e **tanh** — e concatena as representações aprendidas (Fig. 9.1, Algoritmos 9 e 10).

Aqui simplificamos para **um autoencoder de uma camada** por função de ativação (em vez de dois empilhados), o suficiente para ilustrar a ideia central em tempo de aula.


```python
def treinar_autoencoder(X, activation, n_hidden, seed=0):
    """Autoencoder de 1 camada oculta (bloco básico da DAF, Eqs. 9.15-9.18)."""
    ae = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation=activation,
                       solver="adam", max_iter=2000, alpha=1e-4, random_state=seed)
    ae.fit(X, X)  # a saída desejada é a própria entrada -> reconstrução (Eq. 9.17)

    hidden = np.dot(X, ae.coefs_[0]) + ae.intercepts_[0]
    if activation == "logistic":
        encoded = 1 / (1 + np.exp(-hidden))
    else:
        encoded = np.tanh(hidden)
    return encoded


H_sigmoid = treinar_autoencoder(Xs, "logistic", n_hidden=10, seed=0)
H_tanh = treinar_autoencoder(Xs, "tanh", n_hidden=10, seed=1)
H_DAF = np.concatenate([H_sigmoid, H_tanh], axis=1)   # concatenação (Algoritmo 10)

print("Forma da representação DAF (sigmoid + tanh):", H_DAF.shape)

avaliar_classificacao(H_DAF, y, "DAF (10 sigmoid + 10 tanh)")
```

    Forma da representação DAF (sigmoid + tanh): (1000, 20)
    

    DAF (10 sigmoid + 10 tanh)          | F1 = 0.441
    

**Nota pedagógica:** com autoencoders de uma única camada pequena e um dataset sintético pequeno, o DAF simplificado pode não superar a PCA/APCA neste demo — e é esperado. O capítulo justifica o uso de **autoencoders empilhados de 2 camadas** e datasets maiores (ex.: microarray) justamente porque uma única camada rasa nem sempre captura estrutura suficiente. Vale discutir esse ponto em aula como limitação prática dos modelos profundos em cenários de poucos dados.

## 6. Discretização Sensível ao Desbalanceamento: CAIM / ur-CAIM (Seção 9.6)

O critério **CAIM** (Class-Attribute Interdependence Maximization) mede o quão bem um esquema de discretização separa as classes:

$$\text{CAIM}(C,D,A) = \frac{\sum_{r=1}^{n} \max_r^2 / M_{+r}}{m}$$

O capítulo aponta que o CAIM tradicional é **tendencioso para a classe majoritária**; o **ur-CAIM** corrige isso considerando explicitamente os intervalos que contêm exemplos minoritários. Abaixo, uma versão simplificada do critério CAIM compara dois esquemas de discretização para o mesmo atributo.


```python
def caim_criterion(feature, y, boundaries):
    """Critério CAIM simplificado para um esquema de discretização dado."""
    edges = np.sort(boundaries)
    intervals = np.digitize(feature, edges)
    m = intervals.max() + 1
    classes = np.unique(y)
    total = 0.0
    for r in range(m):
        mask = intervals == r
        if mask.sum() == 0:
            continue
        counts = [np.sum(y[mask] == c) for c in classes]
        max_r = max(counts)
        M_plus_r = mask.sum()
        total += (max_r ** 2) / M_plus_r
    return total / m


feat = Xs[:, ranking_relief[0]]  # atributo mais relevante segundo o ReliefF

# Esquema 1: quantis igualitários (ignora a classe minoritária)
quantile_edges = np.quantile(feat, [0.25, 0.5, 0.75])
caim_naive = caim_criterion(feat, y, quantile_edges)

# Esquema 2: cortes calculados a partir da distribuição da classe MINORITÁRIA
minority_vals = feat[y == 1]
custom_edges = np.quantile(minority_vals, [0.15, 0.5, 0.85])
caim_custom = caim_criterion(feat, y, custom_edges)

print(f"CAIM — quantis igualitários (ignora minoria):        {caim_naive:.3f}")
print(f"CAIM — cortes sensíveis à classe minoritária:        {caim_custom:.3f}")
```

    CAIM — quantis igualitários (ignora minoria):        199.269
    CAIM — cortes sensíveis à classe minoritária:        199.820
    

**Leitura:** um esquema de discretização que respeita a distribuição da classe minoritária tende a produzir intervalos mais "puros" (maior CAIM) — a motivação central do ur-CAIM.

## 7. Conclusões (Seção 9.7)

- A redução de dimensionalidade é essencial em cenários desbalanceados de alta dimensão, mas **não pode ser feita de forma ingênua**: técnicas clássicas (SMOTE, PCA, discretização por quantis) tendem a favorecer a classe majoritária.
- **Seleção de atributos** é a abordagem mais estudada — o **ReliefF** e suas variantes (HW, DMR, BMR, Parzen-Relief, MAP-Relief) mostraram-se adaptáveis ao desbalanceamento.
- **Extração de atributos** (linear — APCA — e não linear — autoencoders/DAF) é uma área menos madura, com espaço para novas propostas.
- **Discretização** sensível ao desbalanceamento (ur-CAIM) ainda é pouco explorada e apresenta boas oportunidades de pesquisa.

---
*Notebook preparado como material de apoio à apresentação em sala de aula sobre o Capítulo 9 de "Learning from Imbalanced Data Sets" (Fernández et al., 2018).*
