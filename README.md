# 不规则潮位的调和分解与残差对比

这是独立的气候资料Python项目。`data/`保存合成资料，`climate_data.io`负责严格读取、公历日期和来源字节摘要，`climate_data.domain`提供现有描述统计，`climate_data.harmonic`按实际采样时刻做调和分析（GLS加谐波惩罚、候选评分、冻结外期），`climate_data.cli`将结果输出为JSON。所有观测值、站点、人群、情景与金额都是虚构。字段和研究约定见[资料约定](docs/data-contract.md)，调和分析方法见[方法说明](docs/harmonic-method.md)。

当前Python 3.11及以上可直接使用，没有第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q climate_data tests
python -m climate_data.cli inspect data/normal.json
python -m climate_data.cli analyze data/normal.json
```

`inspect`显示当前原始摘要，`analyze`输出调和分解、候选评分、有效秩与识别诊断、系数三明治协方差、观测时点的趋势/潮汐/残差及冻结外期预测，均不覆盖资料。`invalid-`前缀文件为不合法资料，其他变体使用相同字段保留业务边界。来源字节SHA-256随摘要输出，后续结果应同时标明输入身份、量纲和适用范围。

容器使用固定基础镜像，可运行`docker build -t climate-material .`，随后`docker run --rm climate-material`检查资料。`docker run --rm climate-material python -m unittest discover -s tests -v`执行已有检查。运行不依赖远端账号或实时服务。
