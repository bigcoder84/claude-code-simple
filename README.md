# 简易版ReAct模式ClaudeCode

![](https://image.bigcoder.cn/20251005144939891.gif)

这是一个基于ReAct(Reasoning-Action-Observation)模式的简易代码助手项目。

## 功能特点
- 实现了思考(Reasoning)、行动(Action)、观察(Observation)的循环过程
- 支持通过工具调用获取外部信息
- 可用于代码生成、分析和调试

## 安装方法
1. 克隆本项目
2. 安装依赖：`pip install -r requirements.txt`
3. 创建.env文件，设置OPENAI_API_KEY

## 使用方法

### 同步响应Agent
```shell
python src/agent.py <项目目录>
```

### 流式响应Agent
```shell
python src/agent_stream.py <项目目录>
```
