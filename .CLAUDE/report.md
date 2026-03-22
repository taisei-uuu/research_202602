# ペイロード搬送ドローンスワームの階層型分散安全制御

**Hierarchical Distributed Safe Control for Payload-Carrying Drone Swarms via Affine-Transform Architecture**

---

## 目次

1. [システムのモデル化と定義](#1-システムのモデル化と定義)
2. [制御目的](#2-制御目的)
3. [提案制御則](#3-提案制御則)
   - 3.1 アフィン変換アーキテクチャ
   - 3.2 グラフ構造とGNNポリシー
   - 3.3 名目制御器（LQR + スプリング）
   - 3.4 解析的QPソルバー（安全フィルタ）
   - 3.5 損失関数
   - 3.6 制御フロー全体
4. [シミュレーション検証](#4-シミュレーション検証)
5. [研究背景と既存手法との比較](#5-研究背景と既存手法との比較)

---

## 1. システムのモデル化と定義

### 1.1 システム概要

本研究では，**3機1組のドローンスワームが吊り下げたペイロード（荷物）を搬送しながら，障害物を回避しつつ目標地点へ到達する**問題を扱う．各スワームは，個々のドローン位置ではなく，「重心の並進加速度」と「フォーメーション伸縮加速度」というアフィン変換パラメータで制御される．ケーブルで懸架されたペイロードは非線形振り子として揺動し，その揺れ角は安全制約として課される．

### 1.2 座標系と状態定義

各スワーム $i$（$i = 1, \ldots, n$，$n$ はスワーム数）の状態は以下の3成分で記述される．

**重心状態（CoM state）**：

$$x_i = [p_x,\; p_y,\; v_x,\; v_y]^\top \in \mathbb{R}^4$$

| 変数 | 意味 | 単位 |
|------|------|------|
| $p_x, p_y$ | スワーム重心の2次元位置 | m |
| $v_x, v_y$ | スワーム重心の2次元速度 | m/s |

**スケール状態（Formation scale state）**：

$$s_i = [s,\; \dot{s}]^\top \in \mathbb{R}^2$$

| 変数 | 意味 | 単位 |
|------|------|------|
| $s$ | フォーメーション拡大縮小率（$s=1$ が基準） | 無次元 |
| $\dot{s}$ | スケール変化速度 | 1/s |

**ペイロード状態（Payload swing state）**：

$$\xi_i = [\gamma_x,\; \gamma_y,\; \dot{\gamma}_x,\; \dot{\gamma}_y]^\top \in \mathbb{R}^4$$

| 変数 | 意味 | 単位 |
|------|------|------|
| $\gamma_x, \gamma_y$ | ケーブルの $x$，$y$ 方向の揺れ角 | rad |
| $\dot{\gamma}_x, \dot{\gamma}_y$ | 揺れ角の角速度 | rad/s |

ペイロード状態はGNNへの入力には用いず，QP安全フィルタ（HOCBF）のサイドチャネル情報として使用する．

### 1.3 制御入力

各スワームの制御入力はアフィン変換パラメータの3次元ベクトルである：

$$u_i = [a_{cx},\; a_{cy},\; a_s]^\top \in \mathbb{R}^3$$

| 変数 | 意味 | 単位 |
|------|------|------|
| $a_{cx}, a_{cy}$ | スワーム重心の2次元加速度指令（並進加速度） | m/s² |
| $a_s$ | フォーメーションスケールの加速度指令 | 1/s² |

### 1.4 離散時間力学

タイムステップ $\Delta t$（0.05 s）の離散時間システムとして記述される．

**並進ダイナミクス（Double Integrator）**：

$$p_{t+1} = p_t + v_t \Delta t + \frac{1}{2} \frac{a_{c}}{m} \Delta t^2$$

$$v_{t+1} = \text{clamp}\!\left(v_t + \frac{a_{c}}{m} \Delta t,\; -v_\text{max},\; v_\text{max}\right)$$

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| 質量 | $m$ | 0.1 | kg |
| 最大並進速度 | $v_\text{max}$ | 1.0 | m/s |

**スケールダイナミクス（Double Integrator）**：

$$\dot{s}_{t+1} = \text{clamp}\!\left(\dot{s}_t + a_s \Delta t,\; -\dot{s}_\text{max},\; \dot{s}_\text{max}\right)$$

$$s_{t+1} = \text{clamp}\!\left(s_t + \dot{s}_{t+1} \Delta t,\; s_\text{min},\; s_\text{max}\right)$$

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| スケール下限 | $s_\text{min}$ | 0.4 | — |
| スケール上限 | $s_\text{max}$ | 1.5 | — |
| スケール速度上限 | $\dot{s}_\text{max}$ | 1.0 | 1/s |

**ペイロード揺れダイナミクス（非線形振り子 + 粘性ダンピング）**：

$$\ddot{\gamma}_x = -\frac{g}{l}\sin(\gamma_x) - \frac{a_{cx}}{l}\cos(\gamma_x) - c\,\dot{\gamma}_x$$

$$\ddot{\gamma}_y = -\frac{g}{l}\sin(\gamma_y) - \frac{a_{cy}}{l}\cos(\gamma_y) - c\,\dot{\gamma}_y$$

数値積分は Semi-implicit Euler 法（速度を先に更新してから位置を更新）で行い，数値的安定性を確保する．

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| 重力加速度 | $g$ | 9.81 | m/s² |
| ケーブル長 | $l$ | 1.0 | m |
| 粘性ダンピング係数 | $c$ | 0.03 | 1/s |

### 1.5 フォーメーションジオメトリと動的バウンディングサークル

各スワームは3機のドローンが正三角形フォーメーション（基準外接円半径 $R_\text{form} = 0.5$ m）を形成するものとして，スワーム全体を単一の「動的バウンディングサークル」で近似する：

$$r_\text{swarm}(s) = R_\text{form} \cdot s + r_\text{margin}$$

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| 基準フォーメーション半径 | $R_\text{form}$ | 0.5 | m |
| 安全マージン | $r_\text{margin}$ | 0.2 | m |

$s=1.0$ のとき $r_\text{swarm} = 0.7$ m，$s_\text{min}=0.4$ のとき $0.4$ m，$s_\text{max}=1.5$ のとき $0.95$ m となる．

### 1.6 システムパラメータ一覧

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| タイムステップ | $\Delta t$ | 0.05 | s |
| 環境サイズ | — | 15.0 × 15.0 | m |
| 最大制御入力（並進） | $u_\text{max}$ | 0.3 | N（= kg·m/s²） |
| 通信半径 | $R_\text{comm}$ | 3.0 | m |
| 障害物数 | $n_\text{obs}$ | 6 | — |
| 最大エピソード長 | $T_\text{max}$ | 256 | ステップ |

---

## 2. 制御目的

本システムが達成すべき目標は，以下の3つを同時に満たすことである：

### 2.1 ゴール到達（Task Objective）

各スワーム $i$ が目標位置 $p_\text{goal}^{(i)} \in \mathbb{R}^2$ へ到達すること：

$$\|p_i(T) - p_\text{goal}^{(i)}\| \leq r_\text{goal}$$

ここで $r_\text{goal}$ はゴール判定半径（評価時は 0.3 m を使用）．

### 2.2 安全性（Safety Constraints）

以下の3種類の安全制約をすべての時刻で満たすこと：

**(a) スワーム間衝突回避**：スワーム $i, j$ 間の距離がバウンディングサークルの和より大きいこと：

$$\|p_i - p_j\| \geq r_\text{swarm}(s_i) + r_\text{swarm}(s_j), \quad \forall i \neq j$$

**(b) 障害物衝突回避**：矩形障害物 $k$ からの距離が安全距離以上：

$$d(p_i, \text{Obs}_k) \geq r_\text{swarm}(s_i) + R_\text{obs}, \quad \forall i, k$$

**(c) ペイロード揺れ角制限**：揺れ角が動的上限を超えないこと：

$$|\gamma_x^{(i)}| \leq \gamma_\text{max}(s_i), \quad |\gamma_y^{(i)}| \leq \gamma_\text{max}(s_i), \quad \forall i$$

揺れ角上限 $\gamma_\text{max}(s)$ はスケール $s$ に依存して動的に変化する（詳細は 3.4 節参照）．

### 2.3 スケール制約（Physical Formation Constraint）

フォーメーションスケールが物理的な有効範囲内に収まること：

$$s_\text{min} \leq s_i \leq s_\text{max}, \quad \forall i$$

---

## 3. 提案制御則

### 3.1 アフィン変換アーキテクチャ

#### 基本思想

従来の分散制御では，各ドローンの位置・速度を独立に制御する．本手法では，3機スワームを1つの「アフィン変換可能な剛体」として抽象化し，**並進加速度 $[a_{cx}, a_{cy}]$ とスケール加速度 $a_s$** の3次元制御入力のみで制御する．

この抽象化により：
- 状態空間の次元を大幅に削減（$3 \times 4 = 12$D → $4+2=6$D の主状態）
- フォーメーション形状を明示的に制御変数として扱える
- CBF制約を解析的に定式化しやすい構造が得られる

#### アフィン変換の定義

スワームの各ドローン位置 $q_k$（$k=1,2,3$）は，重心位置 $p_c$ とスケール $s$ を用いて：

$$q_k(t) = p_c(t) + s(t) \cdot \bar{q}_k$$

と表される．ここで $\bar{q}_k \in \mathbb{R}^2$ は基準フォーメーションにおけるドローン $k$ の重心からの相対位置（固定）．制御入力 $a_{cx}, a_{cy}$ は重心の加速度，$a_s$ はスケールの加速度である．

### 3.2 グラフ構造とGNNポリシー

#### グラフの構成

状態を**ヘテロジーニアスグラフ** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ として表現する：

| ノード種別 | 特徴量（3次元 one-hot） | 数 |
|---|---|---|
| Agent（スワーム） | $[1, 0, 0]$ | $n$ |
| Goal（目標位置） | $[0, 1, 0]$ | $n$ |
| Obstacle（LiDARヒット点） | $[0, 0, 1]$ | 可変 |

エッジは通信半径 $R_\text{comm} \cdot s$（スケールに応じて拡張）内にある全ノード対を接続し，エッジ特徴量は：

$$e_{ij} = [\Delta p_x,\; \Delta p_y,\; \Delta v_x,\; \Delta v_y]^\top \in \mathbb{R}^4$$

ここで $\Delta p = p_j - p_i$（相対位置，単位：m），$\Delta v = v_j - v_i$（相対速度，単位：m/s）．

#### GNNアーキテクチャ

GCBF+論文（Zhang et al., 2023）の Eq.18 に基づくAttention付きメッセージパッシングを実装する：

$$h_\theta(z_i) = \psi_4\!\left(\sum_{j \in \tilde{\mathcal{N}}_i} \text{softmax}_j\!\left(\psi_2(q_{ij})\right) \cdot \psi_3(q_{ij})\right)$$

$$q_{ij} = \psi_1\!\left([v_\text{sender};\; v_\text{receiver};\; e_{ij}]\right)$$

各ネットワークの構成：

| ネットワーク | 入力 → 隠れ層 → 出力 | 役割 |
|---|---|---|
| $\psi_1$（エンコーダ） | $(3 \times 2 + 4) \to 256 \to 256 \to 128$ | エッジ特徴量の埋め込み |
| $\psi_2$（Attention） | $128 \to 128 \to 128 \to 1$ | 重要度スコアの計算 |
| $\psi_3$（Value変換） | $128 \to 256 \to 128$ | 集約用特徴量の変換 |
| $\psi_4$（デコーダ） | $128 \to 256 \to 256 \to 3$ | アクション出力 |

全活性化関数はReLU，$\psi_4$ の出力のみ tanh を通して $[-1, 1]^3$ に制限する．

#### ポリシー出力

$$\pi_\phi(\mathcal{G}) = \tanh\!\left(\psi_4(\cdot)\right) \in [-1, 1]^3$$

これを物理スケール（速度オフセット単位）に変換する：

$$\pi_\text{scaled} = \pi_\phi \odot [v_\text{max},\; v_\text{max},\; \dot{s}_\text{max}]$$

ここで $v_\text{max} = 1.0$ m/s，$\dot{s}_\text{max} = 1.0$ s$^{-1}$，$\odot$ は要素積．

### 3.3 名目制御器（LQR + スプリング復元力）

GNNによる学習済みオフセットは，名目制御器の上に「残差補正」として加算される．名目制御器は以下の2段階階層構造を持つ．

#### Level 1：速度指令の生成

目標速度を比例ゲインで生成し，GNNオフセットを加算する：

$$v_\text{ref} = \hat{e} \cdot \min\!\left(v_\text{max},\; K_\text{pos} \cdot \|p_\text{goal} - p\|\right)$$

$$v_\text{target} = v_\text{ref} + \pi_\text{scaled}[:2]$$

ここで $\hat{e} = (p_\text{goal} - p) / \|p_\text{goal} - p\|$ は目標方向の単位ベクトル（無次元），$K_\text{pos} = 0.5$ s$^{-1}$．

#### Level 2：PD追従による加速度生成（並進）

$$a_{c,\text{nom}} = K_v \cdot (v_\text{target} - v_\text{current})$$

$K_v = 2.0$ s$^{-1}$，出力単位は m/s²．

#### スケール制御（スプリング復元力）

スケールは常に $s_\text{max}$ 方向へ引き寄せる「スプリング」で制御する：

$$a_{s,\text{nom}} = k_\text{spring} \cdot (s_\text{max} - s) - c_\text{damp} \cdot \dot{s}$$

| パラメータ | 記号 | 値 | 単位 |
|---|---|---|---|
| スプリング定数 | $k_\text{spring}$ | 0.5 | s$^{-2}$ |
| ダンピング係数 | $c_\text{damp}$ | 0.3 | s$^{-1}$ |

#### 最終名目入力

$$u_\text{nom} = [a_{cx,\text{nom}},\; a_{cy,\text{nom}},\; a_{s,\text{nom}}]^\top \in \mathbb{R}^3$$

$$u_\text{AT} = u_\text{nom} + [0,\; 0,\; \pi_\text{scaled}[2]]^\top$$

ここで $\pi_\text{scaled}[2]$ はGNNによるスケールオフセット（単位：s$^{-1}$，ここでは加速度オフセットとして加算）．

### 3.4 解析的QPソルバー（安全フィルタ）

名目入力 $u_\text{AT}$ をそのまま適用すると安全制約が破られる可能性があるため，各タイムステップで以下の最適化問題（QP）を解いて安全な入力 $u_\text{QP}$ に投影する．

#### 最適化問題の定式化

$$\min_{X \in \mathbb{R}^3,\; \delta \in \mathbb{R}^2} \;\; \frac{1}{2}\|X - u_\text{AT}\|^2 + p \sum_k \delta_k^2$$

subject to:

| 番号 | 制約 | 種別 | 変数 |
|---|---|---|---|
| (1) | ペイロード HOCBF（後述） | ソフト（スラック $\delta$） | $a_{cx}, a_{cy}$ |
| (2) | スケール CBF 下限（後述） | ハード | $a_s$ |
| (3) | スケール CBF 上限（後述） | ハード | $a_s$ |
| (4) | 障害物 CBF（後述） | ハード | $a_{cx}, a_{cy}, a_s$ |
| (5) | 入力飽和 $\|a_c\|_\infty \leq u_\text{max}$ | ハード | $a_{cx}, a_{cy}$ |

スラック重み $p = 100.0$（大きいほどソフト制約を強力に施行）．

#### 制約 (2)(3)：スケール CBF（2次 HOCBF）

スケールの物理的限界 $s \in [s_\text{min}, s_\text{max}]$ を制御バリア関数 $h_s^- = s - s_\text{min}$，$h_s^+ = s_\text{max} - s$ として定式化し，2次 HOCBF の条件を展開すると：

$$a_s \geq -(\alpha_1 + \alpha_2)\dot{s} - \alpha_1\alpha_2(s - s_\text{min}) \quad \text{（下限）}$$

$$a_s \leq -(\alpha_1 + \alpha_2)\dot{s} + \alpha_1\alpha_2(s_\text{max} - s) \quad \text{（上限）}$$

$\alpha_1 = \alpha_2 = 2.0$（クラス $\mathcal{K}$ 関数の係数）．

#### 制約 (4)：障害物 CBF（2次 HOCBF）

スワーム $i$ と障害物 LiDAR ヒット点 $p_\text{obs}^{(k)}$ に対して：

$$h_\text{obs}^{(k)} = \|p_i - p_\text{obs}^{(k)}\|^2 - r_\text{swarm}(s_i)^2$$

$h_\text{obs}$ の1階・2階時間微分を計算し，2次 HOCBF の条件を展開すると：

$$A_\text{obs}^{(k)} \cdot X + \left(\ddot{h}_\text{drift}^{(k)} + (\alpha_1+\alpha_2)\dot{h}^{(k)} + \alpha_1\alpha_2\, h^{(k)}\right) \geq 0$$

$$A_\text{obs}^{(k)} = \left[2\Delta p_x^{(k)},\;\; 2\Delta p_y^{(k)},\;\; -2\,r_\text{swarm}(s)\, R_\text{form}\right] \in \mathbb{R}^{1\times 3}$$

ここで $\Delta p^{(k)} = p_i - p_\text{obs}^{(k)}$ (m)，$\alpha_1 = 0.8$，$\alpha_2 = 0.8$．

#### 制約 (1)：ペイロード揺れ角 HOCBF（動的 $\gamma_\text{max}(s)$）

揺れ角上限をスケールに応じて動的に設定する：

$$\gamma_\text{max}(s) = \arcsin\!\left(\text{clamp}\!\left(\frac{R_\text{form} \cdot s}{l},\; 0,\; 0.95\right)\right)$$

この式はフォーメーション半径 $R_\text{form} \cdot s$ とケーブル長 $l$ の幾何的関係から導出される（スワームが縮小すれば許容揺れ角も小さくなる）．

CBF関数（$x$ 方向の例）：

$$h_x = \gamma_\text{max}(s)^2 - \gamma_x^2$$

これを2段 HOCBF に展開し，制御依存係数と定数項を求めると：

$$C_x = \frac{2\gamma_x \cos(\gamma_x)}{l}, \qquad C_x \cdot a_{cx} + D_x \geq 0 - \delta_x$$

$$D_x = -2\dot{\gamma}_x^2 + \frac{2\gamma_x g}{l}\sin(\gamma_x) + 2\gamma_x c\,\dot{\gamma}_x + (\alpha_1+\alpha_2)(-2\gamma_x\dot{\gamma}_x) + \alpha_1\alpha_2\, h_x$$

$\alpha_1 = \alpha_2 = 2.0$（ペイロード HOCBF 係数）．$y$ 方向も同様．

#### 解法：Dykstra交互射影法

複数の半空間制約への射影を**交互射影（Dykstra-style）**で反復的に解く（デフォルト3反復）：

1. ペイロード HOCBF への射影（最低優先度，ソフト制約）
2. スケール CBF への射影（1次元クランプで解析的）
3. エージェント間衝突回避 CBF への射影
4. 障害物 CBF への射影（最高優先度，最後に適用）
5. 入力飽和クランプ（$|a_c| \leq u_\text{max}$）

半空間 $\{X : a^\top X + b \geq 0\}$ への射影は，違反した場合のみ以下の解析解で計算する：

$$\lambda = \frac{\max(0,\; -(a^\top X + b))}{\|a\|^2 + \epsilon}, \qquad X \leftarrow X + \lambda \cdot a$$

$\epsilon = 10^{-8}$（数値安定化）．全演算はPyTorchテンソル操作によりGPU上でバッチ処理される．

### 3.5 損失関数

学習は **Shuffled Mini-Batch 方式**で行い，以下の損失関数でポリシーネットワーク $\pi_\phi$ を更新する：

$$L = c_\text{prog} \cdot L_\text{progress} + c_\text{arr} \cdot L_\text{arrival} + c_\text{qp} \cdot L_\text{qp} + c_\text{eff} \cdot L_\text{effort}$$

#### $L_\text{progress}$：ゴール距離減少報酬

$$L_\text{progress} = -\mathbb{E}\!\left[\|p_t - p_\text{goal}\| - \|p_{t+1}^\text{pred} - p_\text{goal}\|\right]$$

目標に近づいた量（距離減少）の期待値を最大化する．$p_{t+1}^\text{pred}$ は $v_\text{target}$ から予測した1ステップ先位置であり，GNNへの勾配が流れる（係数 $c_\text{prog} = 1.0$）．

#### $L_\text{arrival}$：到達スパース報酬

$$L_\text{arrival} = -\mathbb{E}\!\left[\mathbf{1}[\|p - p_\text{goal}\| < r_\text{arrival}]\right]$$

ゴール到達（$r_\text{arrival} = 0.3$ m 以内）に対するスパースボーナス（係数 $c_\text{arr} = 5.0$）．勾配は流れない（detach）．

#### $L_\text{qp}$：QP介入ペナルティ

$$L_\text{qp} = \mathbb{E}\!\left[\|u_\text{AT} - u_\text{QP}\|^2\right]$$

QPが名目入力を大きく修正した場合にペナルティを与える．これにより，GNNは**QPが修正を必要としない安全なアクションを事前に学習**するインセンティブを持つ（係数 $c_\text{qp} = 2.0$，単位：(m/s²)²）．

#### $L_\text{effort}$：GNN出力正則化

$$L_\text{effort} = \mathbb{E}\!\left[\pi_\phi^{(x)2} + \pi_\phi^{(y)2} + w_s \cdot \pi_\phi^{(s)2}\right]$$

GNNの出力 $\pi_\phi$ の大きさにペナルティを与え，名目制御器からの逸脱を最小化する（係数 $c_\text{eff} = 0.3$，$w_s = 2.0$）．

### 3.6 制御フロー全体

#### 学習時

```
[Phase 1: データ収集 (torch.no_grad)]
  環境リセット → 各タイムステップ:
    1. グラフ構築 (Agent / Goal / LiDAR-Obstacle ノード)
    2. GNN π_φ(G) → tanh → π_scaled (速度オフセット)
    3. Level 1: v_target = v_ref + π_scaled[:2]
    4. Level 2: u_nom = PD(v_target) + spring(s)
    5. u_AT = u_nom + [0, 0, π_scaled[2]]
    6. QP: u_QP = solve_affine_qp(u_AT, ...) [Dykstra 3反復]
    7. env.step(u_QP) → 次状態
    8. スナップショット保存: (agent, scale, payload, goal, obs, u_AT, u_QP, dist)

[Phase 2: プール構築とシャッフル]
  horizon × batch_size サンプルを i.i.d. として扱いランダムシャッフル

[Phase 3: ミニバッチ学習 (4 epochs)]
  ミニバッチ抽出 → グラフ再構築
  → GNN forward (with grad)
  → u_nom + π → u_AT
  → QP (no_grad)
  → loss = L_progress + L_arrival + L_qp + L_effort
  → Adam.step()
```

#### 推論時（評価・可視化）

```
各タイムステップ:
  1. グラフ構築
  2. π_φ(G) → π_scaled
  3. u_AT = u_nom + GNN offset
  4. u_QP = solve_affine_qp(u_AT)  [安全フィルタ]
  5. env.step(u_QP)
```

#### 学習ハイパーパラメータ

| パラメータ | 値 |
|---|---|
| Optimizer | Adam |
| 学習率 | $3 \times 10^{-4}$ |
| Batch size（環境数） | 128 |
| Horizon（ロールアウト長） | 128 ステップ |
| Mini-batch size | 128 |
| Epochs per step | 4 |
| 勾配クリッピング（max norm） | 2.0 |
| 総学習ステップ数 | 2000〜3000 |

---

## 4. シミュレーション検証

### 4.1 評価設定

評価は `evaluate.py` によるプラグイン方式で行い，以下の3手法を比較する：

| 手法 | アクション生成方法 |
|---|---|
| **Affine Policy（提案手法）** | 学習済みGNN + 解析的QP（Obs-CBF + Scale-CBF + HOCBF） |
| **HOCBF + LQR** | LQR + スプリング + 解析的QP（QP安全フィルタあり，学習なし） |
| **LQR only** | LQR + スプリングのみ（安全フィルタなし，ベースライン） |

各エピソードは $T_\text{max} = 256$ ステップ（12.8 s）で，ランダム生成された環境（エージェント位置，目標位置，障害物配置）で実施される．

### 4.2 評価指標

| 指標 | 定義 | 単位 |
|---|---|---|
| Success Rate | 全エージェントがゴール半径（0.3 m）内に到達したエピソードの割合 | % |
| Safety Rate | 一度も衝突が起きなかったタイムステップの割合 | % |
| $\gamma_\text{mean}$ | 合成揺れ角 $\sqrt{\gamma_x^2 + \gamma_y^2}$ の時間平均 | rad |
| $\gamma_\text{max}$ | 合成揺れ角の最大値 | rad |
| $\gamma_\text{viol}$ | $\gamma > \gamma_\text{max}(s)$ となったタイムステップの割合 | % |
| Scale [min, max] | 観測されたスケール値の範囲 | — |
| Control Effort | $\sum_t \|a_{c,t}\|$ の合計 | m/s² |
| Goal Time | 全エージェントがゴールに到達した最初のタイムステップ | step |

### 4.3 期待される定性的動作

**提案手法（Affine Policy）の期待動作**：

1. **障害物に接近するとスケールが縮小**：Scale-CBF により，$r_\text{swarm}(s)$ が障害物クリアランスに合わせて縮小する．同時に $\gamma_\text{max}(s)$ も縮小するため，ペイロード制御がより慎重になる．

2. **障害物を通過後にスケールが回復**：スプリング制御により $s \to s_\text{max}$ へ復元．フォーメーション拡大中は $\gamma_\text{max}(s)$ も増大し，より速い移動が許容される．

3. **GNNが予測的な経路を学習**：$L_\text{qp}$ ペナルティにより，QPが修正を必要とする場面をGNNが事前に回避するよう学習が進む．

### 4.4 可視化

`visualize.py` により，エピソードの軌跡アニメーションを生成できる：

```bash
python visualize.py --checkpoint ./checkpoints/affine_swarm_1000.pt
```

表示内容：各スワームの重心軌跡，動的バウンディングサークル（スケールに応じて変化），障害物，目標地点，ペイロード揺れ角の時系列グラフ．

---

## 5. 研究背景と既存手法との比較

### 5.1 研究の背景

#### 分散マルチエージェント制御の必要性

複数のエージェントが協調して動作する系（ドローンスワーム，自動運転車群，ロボットチームなど）では，集中制御は通信遅延・スケーラビリティの問題から実用に耐えない．各エージェントが**局所的な観測のみ**から制御則を決定できる**分散制御**が必要となる．

#### 安全制約の課題

強化学習・模倣学習によるエンドツーエンドの学習ベース制御は，高い性能を示す一方で安全性の形式的保証が難しい．一方，制御バリア関数（CBF）に基づく手法は安全性を形式的に保証できるが，複雑な非線形系への適用やスケーラビリティに課題がある．

#### ペイロード搬送の特殊性

ドローンによるペイロード搬送では，ケーブル懸架されたペイロードの揺動が不安定要因となる．特に，フォーメーション変形と揺れ角の間に非自明な結合（スケールが大きいほど揺れが大きくても幾何的に許容できる）があり，この関係を制御設計に陽に組み込む必要がある．

### 5.2 参照論文：GCBF+

本実装の基盤となる参照手法は **GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control**（Zhang et al., 2023, MIT REALM）である．

**GCBF+の主要な貢献**：

| 要素 | 内容 |
|---|---|
| GNN-CBF | グラフニューラルネットワークで CBF 値 $h_\theta(x)$ を学習 |
| 分散実行 | 通信範囲内の近傍エージェントのみの観測で各エージェントが独立に制御 |
| 形式的安全保証 | 学習済みCBFの下で，QP安全フィルタが前向き不変集合を保証 |
| スケーラビリティ | $n=8$ エージェントで学習し $n=1024$ でゼロショット転移 |

**実装フレームワーク**：Jax + Flax，QP ソルバーに JaxProxQP を使用．

### 5.3 本手法の独自性と貢献

本研究は GCBF+ を出発点としながら，**ペイロード搬送という新規タスク**に向けた以下の設計上の貢献を持つ：

#### (a) アフィン変換アーキテクチャ

GCBF+ は各エージェントが個別の位置・速度を持つ等質エージェントを対象とする．本手法は**3機スワームを1つのアフィン変換可能な抽象エージェント**として扱い，状態空間を大幅に削減しながらフォーメーション変形を制御変数に組み込む，新しい抽象化レベルを導入する．

#### (b) 動的ペイロード安全制約（Dynamic HOCBF）

単純な固定揺れ角制限ではなく，フォーメーションスケール $s$ に応じて揺れ角上限 $\gamma_\text{max}(s)$ を動的に変化させる．これにより：
- 障害物回避中（$s$ 小）は厳格なペイロード制約 → 安全優先
- 開けた空間（$s$ 大）は緩和された制約 → 高速移動を許容

という**状況適応的な安全制約**が実現される．

#### (c) 学習不要CBF（Analytical QP）

GCBF+ は CBF 関数そのものをGNNで学習する（safe/unsafe ラベリングが必要）．本手法は CBF を解析的に設計し，**CBF の学習を省略**する．代わりにポリシーネットワーク（$\pi_\phi$）のみを学習し，QP安全フィルタが常に形式的な安全性を保証する構造とした．

| | GCBF+ | 本手法 |
|---|---|---|
| CBF | GNNで学習（safe/unsafe ラベル必要） | 解析的設計（ラベル不要） |
| QP ソルバー | JaxProxQP（外部ライブラリ） | 解析的半空間射影（pure PyTorch） |
| 学習対象 | CBF + Policy（2ネットワーク） | Policy のみ（1ネットワーク） |
| 安全性保証 | 学習CBFの精度に依存 | 解析的CBF → 形式的保証 |
| 新規対応 | 等質エージェント | ペイロード付き異質スワーム |

#### (d) PyTorch 実装と GPU バッチ処理

原論文の Jax 実装を PyTorch で再実装し，解析的 QP ソルバーを純 PyTorch テンソル演算として実装した．これにより：
- 外部 QP ライブラリへの依存なし
- 全 $n$ スワームを単一の GPU カーネル呼び出しで同時処理
- 損失関数との自然な統合（勾配計算との整合性）

### 5.4 既存手法との系統的比較

| 手法カテゴリ | 代表手法 | 安全保証 | 分散実行 | ペイロード | スケーラビリティ |
|---|---|---|---|---|---|
| 集中型MPC | MPC with CBF | 形式的（モデル依存） | × | 困難 | 低（計算量 $O(n^3)$） |
| 学習CBF | GCBF+ | 学習精度依存 | ○ | × | 高（GNNでスケール） |
| 解析CBF + LQR | HOCBF-LQR | 形式的 | ○ | 限定的 | 中 |
| **本手法** | **Affine Policy** | **形式的（解析CBF）** | **○** | **○（動的制約）** | **中〜高** |

### 5.5 今後の発展方向

1. **スワーム数のスケーラビリティ検証**：GNNアーキテクチャにより，少数スワームで学習したポリシーを多数スワームへゼロショット転移する検証（GCBF+が示した $n=8 \to 1024$ のような転移）
2. **3次元拡張**：$z$ 軸方向の高度制御と3次元ペイロード揺動モデルへの拡張
3. **風外乱ロバスト性**：外乱が存在する環境でのロバスト CBF（RCBF）の組み込み
4. **実機検証**：シミュレーションで学習したポリシーの実際のドローンスワームへの Sim-to-Real 転移

---

*本レポートは 2026-03-22 時点のコード実装（ブランチ：main，最終コミット: `4f7a3b2`）に基づいて作成した．*
