# Phase 1 — Research Summary

**Project:** Metro Passenger Forecasting (LSTM + RNN)
**Compiled:** 2026-06-26

This document maps the research landscape relevant to short-term, station-level metro
passenger flow forecasting with recurrent (RNN/LSTM) and spatio-temporal deep learning.
Sources are grouped by theme. Verify exact page numbers/DOIs before formal citation.

---

## A. Foundational deep-learning-for-transport papers

### A1. Lv, Duan, Kang, Li & Wang (2015) — Deep learning for traffic flow (landmark)
- **Title:** Traffic Flow Prediction With Big Data: A Deep Learning Approach
- **Venue/Year:** IEEE Transactions on Intelligent Transportation Systems, 16(2), 865–873, 2015
- **Relevance:** First major use of a deep architecture (stacked autoencoder, greedy layer-wise
  training) for traffic flow features. Establishes that deep models beat classical ML on
  large transport time series — the historical pivot point our project builds on.
- **Link:** https://doi.org/10.1109/TITS.2014.2345663 · TRID: https://trid.trb.org/View/1347910

### A2. Ma, Tao, Wang, Yu & Wang (2015) — First influential LSTM-for-transport
- **Title:** Long Short-Term Memory Neural Network for Traffic Speed Prediction Using Remote
  Microwave Sensor Data
- **Venue/Year:** Transportation Research Part C: Emerging Technologies, 54, 187–197, 2015
- **Relevance:** Directly justifies our LSTM choice. Shows LSTM captures long-span temporal
  dependencies in transport series better than shallow/classical models. Core citation for
  "why LSTM".
- **Link:** https://doi.org/10.1016/j.trc.2015.03.014 · ADS: https://ui.adsabs.harvard.edu/abs/2015TRPC...54..187M/abstract

### A3. Zhao, Chen, Wu, Chen & Liu (2017) — LSTM with temporal–spatial memory grid
- **Title:** LSTM network: a deep learning approach for short-term traffic forecast
- **Venue/Year:** IET Intelligent Transport Systems, 11(2), 68–75, 2017
- **Relevance:** Demonstrates an LSTM network that encodes temporal–spatial correlation via a
  2-D grid of memory units; reinforces LSTM's edge over conventional RNNs for longer horizons.
- **Link:** https://doi.org/10.1049/iet-its.2016.0208

---

## B. Crowd-flow & spatio-temporal deep learning (network structure)

### B1. Zhang, Zheng & Qi (2017) — ST-ResNet (foundational crowd flow)
- **Title:** Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction
- **Venue/Year:** AAAI 2017
- **Relevance:** Canonical inflow/outflow crowd-prediction model. Introduces the
  closeness/period/trend temporal decomposition + external factors (weather, day-of-week) —
  feature-engineering ideas we reuse even in an LSTM prototype.
- **Link:** PDF https://www.zhangjunbo.org/pdf/2017_AAAI_STResNet.pdf · arXiv https://arxiv.org/abs/1610.00081

### B2. Li, Yu, Shahabi & Liu (2018) — DCRNN
- **Title:** Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
- **Venue/Year:** ICLR 2018
- **Relevance:** Combines graph diffusion (spatial) with an encoder–decoder RNN (temporal) and
  scheduled sampling. Reference architecture for the spatio-temporal extension of our work;
  ships METR-LA / PEMS-BAY benchmarks.
- **Link:** arXiv https://arxiv.org/abs/1707.01926 · Code https://github.com/liyaguang/DCRNN

### B3. Yu, Yin & Zhu (2018) — STGCN
- **Title:** Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic
  Forecasting
- **Venue/Year:** IJCAI 2018
- **Relevance:** Pure-convolutional graph model — faster, fewer parameters than RNN graph models.
  Shows the graph route as an alternative to recurrence for capturing station-network topology.
- **Link:** arXiv https://arxiv.org/abs/1709.04875 · Code https://github.com/VeritasYin/STGCN_IJCAI-18

### B4. Wu, Pan, Long, Jiang & Zhang (2019) — Graph WaveNet
- **Title:** Graph WaveNet for Deep Spatial-Temporal Graph Modeling
- **Venue/Year:** IJCAI 2019, 1907–1913
- **Relevance:** Adaptive adjacency (learns station relations, not pre-set) + dilated causal
  convolutions for long sequences. State-of-the-art lineage for the "future work" section.
- **Link:** arXiv https://arxiv.org/abs/1906.00121

---

## C. Metro-specific passenger-flow papers (closest to our task)

### C1. Liu, Liu & Jia (2019) — DeepPF
- **Title:** DeepPF: A deep learning based architecture for metro passenger flow prediction
- **Venue/Year:** Transportation Research Part C: Emerging Technologies, 101, 18–34, 2019
- **Relevance:** The most on-point reference — end-to-end deep model for metro inbound/outbound
  flow that fuses external/environmental factors, temporal dependency, spatial structure and
  metro operational data. Template for our feature design and problem framing.
- **Link:** https://doi.org/10.1016/j.trc.2019.01.027 · https://www.sciencedirect.com/science/article/abs/pii/S0968090X18306806

### C2. Hao, Lee & Zhao (2019) — Seq2Seq + Attention for metro
- **Title:** Sequence to sequence learning with attention mechanism for short-term passenger flow
  prediction in large-scale metro system
- **Venue/Year:** Transportation Research Part C, 107, 287–300, 2019
- **Relevance:** Multi-step, all-station prediction with encoder–decoder + attention. Defines
  the "multi-horizon" framing and shows attention's gain on long-range dependence — a natural
  upgrade path from a plain LSTM.
- **Link:** https://doi.org/10.1016/j.trc.2019.08.005 · https://www.sciencedirect.com/science/article/abs/pii/S0968090X19300245

### C3. Wang et al. (2023) — Clustering + Deep Learning for NEW stations
- **Title:** Short-Term Inbound and Outbound Passenger Flow Prediction for New Metro Stations
  Based on Clustering and Deep Learning (K-means + Sp-LSTM + MAE feedback)
- **Venue/Year:** Journal of Advanced Transportation, 2023
- **Relevance:** Tackles the cold-start challenge (new stations, little history) called out in
  the brief; useful for the "challenges" discussion.
- **Link:** https://doi.org/10.1155/2023/6659916

### C4. PSO-LSTM on AFC data (2022)
- **Title:** Short-Term Subway Inbound Passenger Flow Prediction Based on AFC Data and PSO-LSTM
  Optimized Model
- **Venue/Year:** Urban Rail Transit (Springer), 2022
- **Relevance:** Concrete AFC→LSTM pipeline with hyper-parameter optimization (PSO); confirms
  AFC tap data as the primary signal and gives a baseline-tuning idea.
- **Link:** https://doi.org/10.1007/s40864-022-00166-x

### C5. CNN-LSTM metro station model (2023)
- **Title:** Prediction of Short-term Passenger Flow in the Metro Station with CNN-LSTM Model
- **Venue/Year:** IEEE Conference Publication, 2023
- **Relevance:** Shows the common CNN-LSTM hybrid (CNN for local spatial/temporal feature
  extraction feeding LSTM) — a strong, implementable second model for our notebook.
- **Link:** https://ieeexplore.ieee.org/document/10176978/

### C6. Multi-source big-data ST-LSTM (2025)
- **Title:** Short-term passenger flow prediction for urban rail systems: A deep learning approach
  utilizing multi-source big data (ST-LSTM)
- **Venue/Year:** PLOS One, 2025
- **Relevance:** Recent multi-source ST-LSTM with temporal + spatial + fusion modules; current
  state-of-practice framing and open-access methodology detail.
- **Link:** https://doi.org/10.1371/journal.pone.0333094 · PMC https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12500114/

---

## D. Datasets & benchmarks

### D1. Hangzhou Metro AFC dataset (Tianchi competition) — PRIMARY candidate
- **What:** ~70M tap-in/tap-out AFC records, 80–81 stations, 3 lines, 1–25 Jan 2019;
  pre-aggregated 10-minute station flow available. MAE was the official metric.
- **Relevance:** The de-facto public benchmark for metro flow; perfect fit for short-term,
  station-level inflow/outflow at 10-min resolution.
- **Links:** Zenodo mirror https://zenodo.org/records/3145404 · IEEE DataPort
  https://ieee-dataport.org/documents/hangzhou-metro-passenger-flow-dataset ·
  Sample solution repo https://github.com/shiwang0211/passenger_flow_prediciton

### D2. METR-LA / PEMS-BAY (traffic, via DCRNN)
- **What:** Standard spatio-temporal graph benchmarks (speeds). Background/transfer reference.
- **Link:** https://github.com/liyaguang/DCRNN

### D3. TaxiBJ / BikeNYC (crowd flow grids, via ST-ResNet)
- **What:** Citywide inflow/outflow grids — relevant background for crowd-flow framing.
- **Link:** https://github.com/lucktroy/DeepST

---

## E. Classical baselines (for comparison context)
- **ARIMA/SARIMA & exponential smoothing** — pre-deep-learning standard for short-term transit
  forecasting; used as baselines to show LSTM's gain.
- **SVR / kNN / Random Forest** — classical ML baselines on flow prediction.
- (No single canonical paper required; cite a survey such as the deep-learning traffic-flow
  review: https://www.tandfonline.com/doi/full/10.1080/23311916.2021.2010510)

---

## F. Synthesis — what the literature tells us for our prototype
1. **LSTM is the well-validated core** for short-term metro flow (A2, A3, C1–C6).
2. **Feature engineering matters as much as architecture:** calendar/cyclic features, recent-lag
   "closeness", weekly "period", and external factors (weather/events) recur across ST-ResNet,
   DeepPF and ST-LSTM.
3. **Natural framing for a demonstrable prototype:** short-term, station-level inbound/outbound
   flow at 10–15 min resolution on the Hangzhou Metro AFC dataset, LSTM (+ optional CNN-LSTM/
   attention variant), evaluated with MAE/RMSE/MAPE/R².
4. **Upgrade paths** (future work): attention seq2seq (C2), graph models DCRNN/STGCN/Graph
   WaveNet (B2–B4) to capture inter-station spatial coupling.
