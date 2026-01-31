import ast
import inspect
import os
import re
import sys
import threading
import time
from string import Template
from typing import List, Callable, Tuple, Optional, Generator

import click
from dotenv import load_dotenv
from openai import OpenAI
import platform

from prompt_template import react_system_prompt_template


class Spinner:
    """简单的旋转进度指示器"""

    def __init__(self, message: str = "Processing"):
        self.message = message
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._stop_event = threading.Event()

    def _spin(self):
        """内部旋转循环"""
        while not self._stop_event.is_set():
            for frame in self.frames:
                if self._stop_event.is_set():
                    break
                sys.stdout.write(f"\r{frame} {self.message}")
                sys.stdout.flush()
                time.sleep(0.08)
        # 清除 spinner
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()

    def start(self):
        """启动 spinner"""
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        """停止 spinner"""
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=0.2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class Colors:
    """终端颜色常量"""

    @staticmethod
    def cyan(text: str) -> str:
        return f"\033[36m{text}\033[0m"

    @staticmethod
    def green(text: str) -> str:
        return f"\033[32m{text}\033[0m"

    @staticmethod
    def yellow(text: str) -> str:
        return f"\033[33m{text}\033[0m"

    @staticmethod
    def blue(text: str) -> str:
        return f"\033[34m{text}\033[0m"

    @staticmethod
    def red(text: str) -> str:
        return f"\033[31m{text}\033[0m"

    @staticmethod
    def dim(text: str) -> str:
        return f"\033[2m{text}\033[0m"

    @staticmethod
    def bold(text: str) -> str:
        return f"\033[1m{text}\033[0m"


class StreamingReActAgent:
    """支持流式输出的 ReAct Agent"""

    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=self._get_api_key(),
        )
        self.colors = Colors()

    def run(self, user_input: str):
        """运行 agent，使用流式输出"""
        messages = [
            {"role": "system", "content": self._render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        step_count = 0
        while True:
            step_count += 1

            # 显示步骤分隔符
            print(f"\n{self.colors.dim('─' * 60)}")
            print(f"{self.colors.dim(f'Step {step_count}')}")
            print(f"{self.colors.dim('─' * 60)}")

            # 流式请求模型
            content = self._call_model_streaming(messages)

            if content is None:
                break

            # 解析并显示 Thought
            thought = self._extract_tag(content, "thought")
            if thought:
                print(f"\n{self.colors.blue('💭 Thought')} {thought}")

            # 检查是否有 Final Answer
            if "<final_answer>" in content:
                final_answer = self._extract_tag(content, "final_answer")
                print(f"\n{self.colors.green('✅ Final Answer')} {final_answer}")
                return final_answer

            # 解析 Action
            action = self._extract_tag(content, "action")
            if not action:
                print(f"{self.colors.yellow('⚠️  模型未输出 <action> 标签')}")
                break

            tool_name, args = self._parse_action(action)

            # 显示 Action
            args_str = ", ".join(repr(arg) for arg in args)
            print(f"\n{self.colors.cyan('🔧 Action')} {self.colors.bold(tool_name)}({args_str})")

            # 确认执行
            should_continue = input(f"\n{self.colors.dim('继续执行？(Y/N): ')} ") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print(f"{self.colors.yellow('操作已取消')}")
                return "操作被用户取消"

            # 执行工具并显示进度
            print(f"\n{self.colors.dim('⏳ 正在执行...')}")

            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"

            print(f"\n{self.colors.yellow('🔍 Observation')} {observation[:500]}{'...' if len(observation) > 500 else ''}")

            # 将 observation 添加到消息历史
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

    def _call_model_streaming(self, messages: list) -> Optional[str]:
        """流式调用模型，实时显示输出"""
        full_content = ""
        current_tag = ""
        in_tag = False
        tag_name = ""
        tag_content = []

        print(f"{self.colors.dim('🤖 Assistant: ')}", end="", flush=True)

        try:
            with Spinner("Thinking"):
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # 检测标签开始
                    if delta.content and "<" in delta.content:
                        # 检查是否是标签开始
                        for tag in ["<thought>", "<action>", "<final_answer>"]:
                            if tag in delta.content or (full_content + delta.content).endswith(tag[:-1]):
                                in_tag = True
                                tag_name = tag[1:-1]  # 去掉 <>
                                # 打印之前累积的内容
                                if full_content:
                                    # 清除 spinner 并显示内容
                                    print(f"\r{' ' * 30}\r", end="")
                                    print(self.colors.dim(full_content.replace(tag, "")), end="", flush=True)
                                    full_content = ""
                                break

                    full_content += (delta.content or "")

                    # 如果在标签内，累积标签内容
                    if in_tag:
                        if delta.content:
                            tag_content.append(delta.content)

                        # 实时显示标签内容（去除标签标记）
                        display_content = "".join(tag_content)
                        display_content = re.sub(rf'<{tag_name}>', '', display_content)
                        display_content = re.sub(rf'</{tag_name}>.*', '', display_content)

                        # 清除 spinner 并显示实时内容
                        sys.stdout.write(f"\r{' ' * 30}\r")
                        if tag_name == "thought":
                            sys.stdout.write(f"{self.colors.dim(display_content)}")
                        elif tag_name == "action":
                            sys.stdout.write(f"{self.colors.cyan(display_content)}")
                        elif tag_name == "final_answer":
                            sys.stdout.write(f"{self.colors.green(display_content)}")
                        sys.stdout.flush()

                        # 检测标签结束
                        if f"</{tag_name}>" in "".join(tag_content):
                            in_tag = False
                            tag_name = ""
                            tag_content = []
                            print()  # 换行
                    elif delta.content and not any(t in delta.content for t in ["<thought>", "<action>", "<final_answer>"]):
                        # 不在标签内时，显示普通内容（灰色）
                        sys.stdout.write(f"\r{' ' * 30}\r")
                        sys.stdout.write(self.colors.dim(delta.content))
                        sys.stdout.flush()

            print()  # 确保最后换行

            # 将完整回复添加到消息历史
            messages.append({"role": "assistant", "content": full_content})
            return full_content

        except Exception as e:
            print(f"\n{self.colors.red(f'❌ 模型调用错误: {str(e)}')}")
            return None

    def _extract_tag(self, content: str, tag_name: str) -> Optional[str]:
        """从内容中提取指定标签的内容"""
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        """解析 action 字符串，提取函数名和参数"""
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid function call syntax: {code_str}")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i - 1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
                (arg_str.startswith("'") and arg_str.endswith("'")):
            inner_str = arg_str[1:-1]
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str

        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            return arg_str

    def _get_tool_list(self) -> str:
        """生成工具列表字符串"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def _render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板"""
        tool_list = self._get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            operating_system=self._get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list,
            project_directory=self.project_directory
        )

    @staticmethod
    def _get_api_key() -> str:
        """从环境变量加载 API key"""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key

    def _get_operating_system_name(self) -> str:
        """获取操作系统名称"""
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }
        return os_map.get(platform.system(), "Unknown")


# ========== 工具函数定义 ==========

def read_file(file_path: str) -> str:
    """读取文件内容

    Args:
        file_path: 文件的绝对路径

    Returns:
        文件的文本内容
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_to_file(file_path: str, content: str) -> str:
    """将内容写入文件

    Args:
        file_path: 文件的绝对路径
        content: 要写入的内容

    Returns:
        操作结果消息
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"成功写入 {file_path}"


def run_terminal_command(command: str) -> str:
    """执行终端命令

    Args:
        command: 要执行的 shell 命令

    Returns:
        命令执行结果
    """
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout if result.stdout else "执行成功（无输出）"
    else:
        return f"命令执行失败（退出码 {result.returncode}）: {result.stderr}"


def list_directory(directory: str = ".") -> str:
    """列出目录内容

    Args:
        directory: 目录路径，默认为当前目录

    Returns:
        目录内容列表
    """
    try:
        items = os.listdir(directory)
        return "\n".join(items)
    except Exception as e:
        return f"无法列出目录: {str(e)}"


# ========== CLI 入口 ==========

@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(project_directory: str):
    """ReAct Agent Demo - 支持流式输出的智能助手"""
    project_dir = os.path.abspath(project_directory)

    # 定义可用工具
    tools = [
        read_file,
        write_to_file,
        run_terminal_command,
        list_directory,
    ]

    # 创建 agent
    agent = StreamingReActAgent(
        tools=tools,
        model="deepseek-chat",
        project_directory=project_dir
    )

    print(f"\n{agent.colors.bold('🚀 ReAct Agent Demo')}")

    work_dir_msg = f"工作目录: {project_dir}"
    print(f"{agent.colors.dim(work_dir_msg)}")

    quit_msg = '输入 "quit" 或 "exit" 退出'
    print(f"{agent.colors.dim(quit_msg)}")

    print(f"{agent.colors.dim('─' * 60)}\n")

    while True:
        try:
            task = input(f"{agent.colors.green('🤔 您的任务: ')} ").strip()

            if not task:
                continue

            if task.lower() in ['quit', 'exit', 'q']:
                print(f"{agent.colors.dim('👋 再见！')}")
                break

            agent.run(task)

        except KeyboardInterrupt:
            print(f"\n{agent.colors.yellow('⚠️  操作已取消')}")
        except EOFError:
            print(f"\n{agent.colors.dim('👋 再见！')}")
            break
        except Exception as e:
            print(f"\n{agent.colors.red(f'❌ 错误: {str(e)}')}")


if __name__ == "__main__":
    main()
