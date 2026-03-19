import shutil
from pathlib import Path

def replace_txt(source: str):
    source_path = f"C://Users//li.zhengen//Desktop//main-dis//drivecycle//{source}.txt"
    target_path = "C://Users//li.zhengen//Documents//MWORKS//mw_mol//TAEconomy//TAEconomy//Resources//DriveCycles//CLTCP.txt"
    try:
        shutil.copy2(source_path, target_path)
        print(f"✅ 替换成功")
        return True
    except Exception as e:
        print(f"❌ 替换失败：{e}")
        return False


# ==================== 使用示例 ====================
if __name__ == "__main__":
    replace_txt('CLTCP')
    replace_txt('WLTC')