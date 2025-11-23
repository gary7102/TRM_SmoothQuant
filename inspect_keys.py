import torch
import sys
import os

# 請確認這個路徑正確
CHECKPOINT_PATH = "checkpoints/sudoku_att/step_21700"

def inspect():
    print(f"正在檢查: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("錯誤: 找不到檔案")
        return

    try:
        # 載入 checkpoint
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
        # 判斷結構
        state_dict = None
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                print("偵測到: 包含 'state_dict' 的字典結構")
                state_dict = ckpt["state_dict"]
            else:
                print("偵測到: 直接的 state_dict 結構")
                state_dict = ckpt
        else:
            print("未知結構")
            return

        # 印出前 20 個 keys
        keys = list(state_dict.keys())
        print(f"\n總共有 {len(keys)} 個參數。前 20 個 keys:")
        for k in keys[:20]:
            print(f"  {k}")

        # 搜尋關鍵字
        print("\n搜尋 'embed' 相關的 keys:")
        embed_keys = [k for k in keys if "embed" in k]
        for k in embed_keys:
            print(f"  {k}")

        print("\n搜尋 'H_init' 相關的 keys:")
        h_keys = [k for k in keys if "H_init" in k]
        for k in h_keys:
            print(f"  {k}")

    except Exception as e:
        print(f"讀取失敗: {e}")

if __name__ == "__main__":
    inspect()
