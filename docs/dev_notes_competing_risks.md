# 竞争风险区间概率：实现与复跑说明

本说明对应在 Statsmodels 0.14.5 源码上新增的两块能力：

1. `PHReg` 的**事件后（右连续）口径**：
   - `PHReg.baseline_cumulative_hazard(params, post_event=False)`
   - `PHReg.baseline_cumulative_hazard_function(params, post_event=False)`
   - `PHReg.predict(..., post_event=False)` 与
     `PHRegResults.predict(..., post_event=False)` 中 `pred_type` 为
     `surv` / `cumhaz` 的分支。
   - 省略 `post_event` 时完全沿用历史的左连续行为。
2. `CumIncidenceRight` 的**按起点/时点查询**与
   `CumIncidenceRight.from_phreg(fits)` 拟合模式。

所有概率与协方差运算使用 NumPy `float64`；生存函数沿用
`scipy.interpolate` / `scipy.stats` 中既有的阶跃与分布工具；分析表
用 pandas 组织，查询结果返回 `pandas.DataFrame`。

## 环境

- Python 3.11
- 离线可复跑；本仓库以 editable 方式安装到独立虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install numpy scipy pandas patsy cython packaging \
    setuptools wheel setuptools-scm pytest flake8
.venv/bin/pip install --no-build-isolation -e .
```

## 可复跑的测试命令

### 1. 整个 duration 模块的单元测试（含新增测试与旧接口回归）

```bash
.venv/bin/python -m pytest statsmodels/duration/tests/ -q
```

新增的两个测试文件：

- `statsmodels/duration/tests/test_cumincidence_query.py`
  经验曲线查询、事件边界、区间拼接、非法输入、无可存活质量、旧属性回归，
  以及 PHReg 事件后基准风险/生存口径。
- `statsmodels/duration/tests/test_cumincidence_phreg.py`
  Breslow/Efron 两种 ties、同日多故障、分层且某层缺一种故障、
  延迟入组、预测 offset、单结局退化、标签重命名、队列冲突、
  包装结果与正则化拒绝。

### 2. 独立的手工小样本参考（不依赖被测实现给定期望值）

```bash
.venv/bin/python -I dev_scripts/hand_reference.py
```

期望值由 Nelson-Aalen / Kaplan-Meier 定义在极小确定性样本上手算得到，
覆盖事件边界、区间拼接恒等式、单结局退化与概率闭合。

### 3. 时间与额外内存规模证据

```bash
.venv/bin/python -I dev_scripts/scale_benchmark.py
```

## 实际输出（本次开发环境）

### 单元测试

```
110 passed, 1 skipped, 11 warnings in 1.60s
```

其中 1 个 skip 与 11 个 warning 均为仓库既有现象（绘图后端、
pandas/scipy 的弃用告警，以及经验 CIF 标准误公式里既有的
除零 RuntimeWarning），与本次改动无关。

### 手工参考

```
all hand-reference checks passed
```

### 规模证据（K=4 个故障，连续稀疏事件日，6 个查询时点含 1 个重复）

```
      n      R      M   R*M*K floats  R*M*K MB   peak MB    out MB      sec
    500    500    326          652,000       5.22      1.88     0.12    0.133
   2000   2000   1418       11,344,000      90.75      7.38     0.48    0.589
   8000   8000   5589      178,848,000    1430.78     29.49     1.92    3.459

peak bytes per query row: [3754.2 3689.6 3686.6]
sec per 1000 query rows: [0.2663 0.2945 0.4324]
```

读法：

- 若真的物化 `R × M × K` 数组，8000 行需要约 1.43 GB；实测查询峰值
  约 29.5 MB，且**每查询行的峰值字节稳定在约 3.7 KB**，不随 R 累积，
  说明工作区里不存在 `预测行 × 全部事件时点 × 故障数` 的数组。
- 单行的临时数组规模是 `O(M·K)`（按层合并网格 × 故障数），逐行处理、
  用完即释放；唯一随 R 增长的是最终结果表 `R × len(times) × (K+1)`。
- 时间复杂度为 `O(R · M · K)`（每行在该层合并网格上做向量化的
  累积与按比例分配），实测秒/千行随 M 缓增；查询时点用
  `searchsorted` 二分，时点数量本身只贡献 `O(R · T · log M)`。
- 共有事件时点按**层**取并集合并为一张网格（`np.unique`），
  而不是按 `(行, 时点, 故障)` 扩张；稀疏、错开的故障日期下 M 只随
  实际 distinct 事件数增长。

## 同日多故障的分配规则

在共享事件时点 t，当天流失的无故障概率质量为
`S(t-) · (1 - exp(-Σ_k ΔH_k(t)))`，按各故障**缩放后风险跳量**
`ΔH_k(t)` 的比例分配：

```
mass_k(t) = 当天总流失质量 × ΔH_k(t) / Σ_j ΔH_j(t)
```

性质：对标签对称；相同跳量分到相同概率；零跳量分不到概率；
各故障概率加剩余生存概率严格等于 1。

隐含假设与局限：该规则等价于 Aalen-Johansen 对同日 ties 的标准处理，
假定同日并列事件在无穷小的时间排序下、各原因特异性危害在日内为
独立常数危害。若事件时间被 coarsen（真实有先后但只记录到天），
日内的拆分无法从数据本身识别；换更细的时间粒度会改变"哪些事件
同日"，从而改变日内分配（当天总流失质量本身不变）。

## 关键口径与边界约定

- 事件后口径：首个事件以前为 0；事件当天包含当天的累积风险跳变；
  最后一次跳变计入其后所有时间。
- 起点 `origin` 视为已无故障经过；只统计 `(origin, time]` 内事件。
  `time == origin` 时各故障概率为 0、无故障概率为 1。
- 分层：某已知层没有某种故障的事件时，该原因在这一层风险恒为 0；
  查询给出未知层标签会报错。
- 经验曲线首版仅支持未加权、无 exog 调整；带权或 exog 调整时查询
  明确报错。起点已无可存活质量（S=0）时拒绝查询。
- 单结局（`from_phreg` 只给一个映射）时，区间条件无故障概率与
  `PHReg.predict(..., pred_type="surv", post_event=True)` 的
  `S(t)/S(origin)` 比值一致。

## 未验证 / 未覆盖项（如实列出）

- 未做参数估计不确定性的传播：查询返回点概率，不给 CIF 的置信区间
  或标准误（经验构造路径仍保留原有的 `cinc_se`，拟合路径 `cinc_se`
  为 `None`）。
- 经验模式的区间查询不支持频率权重、exog 核调整（按需求首版明确
  报错而非近似）。
- 未在真实大规模设备随访数据上端到端跑业务报表；规模证据来自
  生成的稀疏竞争风险数据（`scale_benchmark.py`）。
- 文档中的 ipython 指令块未在本机完整构建 HTML 文档验证
  （本机未安装完整 Sphinx 文档工具链）；代码与 docstring 已通过
  pytest 与 flake8。
- 顶层 `statsmodels.api` 在本机较新的 numpy/pandas 组合下存在与本次
  改动无关的既有导入问题（`deprecate_kwarg` 签名），因此验证脚本均
  直接从 `statsmodels.duration...` 子模块导入，并以 `python -I`
  隔离运行。
