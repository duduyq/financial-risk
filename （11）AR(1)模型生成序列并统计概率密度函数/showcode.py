# -*- coding: utf-8 -*-
"""显示某个源代码文件的内容（供演示时在命令框里展示代码用）"""
import sys

if len(sys.argv) < 2:
    print("用法: python showcode.py <文件名>")
    sys.exit(1)

with open(sys.argv[1], encoding="utf-8") as f:
    print(f.read())
