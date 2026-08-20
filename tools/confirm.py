# 写入/编辑/删除类操作前的终端阻塞式人工审核

"""
所有会修改或删除文件的工具（write_file / edit_file / move_file /
create_directory / python_exec / run_python_script）都要在真正执行前
调用 confirm_action，打印操作摘要并阻塞等待用户输入 y/n。
"""


def confirm_action(summary: str) -> bool:
    """打印操作摘要，阻塞等待用户输入 y/n 确认。"""
    print("\n" + "=" * 60)
    print("[审核] 即将执行以下操作，需要你确认：")
    print(summary)
    print("=" * 60)
    while True:
        choice = input("是否允许执行？(y=允许 / n=拒绝): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("请输入 y 或 n。")
