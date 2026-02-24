import os

# 要遍历的目录
root_dir = "/Users/lixiaojie/Downloads/U-MARVEL/scripts/"

# 需要替换的内容
replacements = {
    "lamra_npu": "u-marvel",
    "/group/40077/users/clearli/": "/home/user_name/"
}

def is_text_file(filepath):
    """
    简单判断是否为文本文件（避免修改二进制文件）
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except:
        return False


for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(root, file)

        if not is_text_file(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = content
            for old, new in replacements.items():
                new_content = new_content.replace(old, new)

            # 如果有修改才写回
            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated: {file_path}")

        except Exception as e:
            print(f"Skip {file_path}, error: {e}")