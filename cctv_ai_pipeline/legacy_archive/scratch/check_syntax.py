import re

html_path = r"E:\AIVLE_10team\results\cctv_simulation_dashboard.html"
with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# script 블록 추출
scripts = re.findall(r"<script>(.*?)</script>", content, re.DOTALL)
for i, script in enumerate(scripts):
    print(f"--- Script {i} Syntax Check ---")
    stack = []
    lines = script.split('\n')
    for line_num, line in enumerate(lines, 1):
        for col_num, char in enumerate(line, 1):
            if char in '{[(':
                stack.append((char, line_num, col_num))
            elif char in '}])':
                if not stack:
                    print(f"Unmatched closing char '{char}' at line {line_num}, col {col_num}: {line.strip()}")
                else:
                    opening, op_line, op_col = stack.pop()
                    if (opening == '{' and char != '}') or \
                       (opening == '[' and char != ']') or \
                       (opening == '(' and char != ')'):
                        print(f"Mismatched char: opened '{opening}' at line {op_line}:{op_col}, closed '{char}' at line {line_num}:{col_num}")
                        print(f"  Open line: {lines[op_line-1].strip()}")
                        print(f"  Close line: {line.strip()}")
    if stack:
        print(f"Unclosed braces left at the end of script:")
        for op, op_line, op_col in stack:
            print(f"  '{op}' opened at line {op_line}:{op_col} -> {lines[op_line-1].strip()}")
    else:
        print("No brace mismatch found.")
