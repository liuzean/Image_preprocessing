import os
import win32com.client

def check_and_update_schedule():
    """
    检查schedule.xlsx文件中A列有内容的行，根据D列是否有内容进行红色填充
    """
    file_path = r"E:\研究生资料\schedule.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
        return
    
    # 连接到已经打开的 Excel 程序，如果没打开则创建一个新实例
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
    except Exception:
        excel = win32com.client.Dispatch("Excel.Application")
    
    # 确保Excel运行窗口在前台可见
    excel.Visible = True
    
    # 检查工作簿是否目前已经被打开（避免只读冲突）
    wb = None
    for w in excel.Workbooks:
        # 将路径做格式化并转小写匹配（消除盘符大小写或斜杠差异）
        if os.path.normpath(w.FullName).lower() == os.path.normpath(file_path).lower():
            wb = w
            break
    
    # 如果没打开工作簿，正常去打开它
    if not wb:
        wb = excel.Workbooks.Open(file_path)
    
    ws = wb.Sheets(1)  # 使用第一个sheet
    
    # 通过查找A列判断最后一行
    # -4162 是 Excel VBA 里的常量 xlUp
    last_row = ws.Cells(ws.Rows.Count, 1).End(-4162).Row
    
    # 定义红色填充颜色 (RGB: 255, 0, 0)
    red_color = 255  # Excel中红色的颜色索引
    
    row_count = 0
    updated_rows = []
    
    # 遍历所有行
    for r in range(1, last_row + 1):
        cell_a = ws.Cells(r, 1)
        cell_d = ws.Cells(r, 4)
        
        # 检查A列是否有内容
        a_value = cell_a.Value
        if a_value is not None and str(a_value).strip() != "":
            row_count += 1
            
            # 检查D列是否有内容
            d_value = cell_d.Value
            d_has_content = d_value is not None and str(d_value).strip() != ""
            
            # 检查D列是否有红色填充（通过ColorIndex或Interior.Color）
            d_is_red = False
            try:
                # 检查标准红色（RGB: 255, 0, 0）
                if cell_d.Interior.Color == 255:  # 纯红色
                    d_is_red = True
            except:
                pass
            
            # 根据条件进行操作
            if not d_has_content and not d_is_red:
                # D列为空且没有红色填充，则填充红色
                cell_d.Interior.Color = 255  # 设置为红色
                updated_rows.append((r, "填充红色"))
            
            elif d_has_content and d_is_red:
                # D列有内容且有红色填充，则移除填充
                cell_d.Interior.ColorIndex = -4142  # xlNone 清除填充
                updated_rows.append((r, "移除填充"))
    
    # 保存工作簿
    try:
        wb.Save()
    except Exception as e:
        print(f"错误：无法保存文件 - {e}")
        return
    
    # 打印总结
    print(f"\n处理完成:")
    print(f"  扫描行数：{row_count} 行（A列有内容）")
    print(f"  更新行数：{len(updated_rows)} 行")
    if updated_rows:
        for row_num, action in updated_rows:
            print(f"    第 {row_num} 行：{action}")


if __name__ == "__main__":
    check_and_update_schedule()
