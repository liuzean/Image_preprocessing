import os
import win32com.client
from datetime import datetime, timedelta

def parse_excel_date(date_val, date_text):
    """解析Excel中的日期"""
    # 如果已经被Excel隐式识别为日期类型（在win32com中带时区的datetime对象）
    if hasattr(date_val, 'year') and hasattr(date_val, 'month'):
        return datetime(date_val.year, date_val.month, date_val.day)
        
    # 获取单元格上的文本或值（兼容常规格式）
    date_str = str(date_text).strip()
    if not date_str or date_str == 'None':
        date_str = str(date_val).strip()

    base_year = datetime.now().year
    
    # 提取多天或特殊格式 （例如：6.8-6.10, 或者 5.30-31）
    if "-" in date_str:
        parts = date_str.split("-")
        last_part = parts[-1]
        first_part = parts[0]
        
        # 判断后一半有没有月份 (如 6.10) 还是只有日子 (如 31)
        if "." in last_part:
            m, d = last_part.split(".")
        else:
            m = first_part.split(".")[0]
            d = last_part
    else:
        # 常规格式诸如 6.4
        m, d = date_str.split(".")
        
    return datetime(base_year, int(m), int(d))


def main():
    file_path = r"E:\研究生资料\schedule.xlsx"
    
    # 连接到已经打开的 Excel 程序，如果没打开则创建一个新实例
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
    except Exception:
        excel = win32com.client.Dispatch("Excel.Application")
    
    # 确保Excel运行窗口在前台可见，而且运行完以后不关闭
    excel.Visible = True
    
    # 检查工作薄是否目前已经被打开（避免只读冲突）
    wb = None
    for w in excel.Workbooks:
        # 将路径做格式化并转小写匹配（消除盘符大小写或斜杠差异）
        if os.path.normpath(w.FullName).lower() == os.path.normpath(file_path).lower():
            wb = w
            break
            
    # 如果没打开工作簿，正常去打开它
    if not wb:
        if not os.path.exists(file_path):
            print(f"错误: 找不到指定的文件 -> {file_path}")
            return
        wb = excel.Workbooks.Open(file_path)
        
    ws = wb.Sheets("Sheet1")
    
    # 通过查找A列判断目前填写的最后一行，并且确认新的起始空行
    # -4162 是 Excel VBA 里的常量 xlUp
    last_row = ws.Cells(ws.Rows.Count, "A").End(-4162).Row
    next_row = last_row + 1
    
    # 从目前最后一行开始“向上”倒推遍历，寻找A列为“平复”的最下方的一行
    last_date = None
    for r in range(last_row, 0, -1):
        if ws.Cells(r, 1).Value == "平复":
            val_b = ws.Cells(r, 2).Value
            text_b = ws.Cells(r, 2).Text
            try:
                last_date = parse_excel_date(val_b, text_b)
            except Exception as e:
                print(f"解析第 {r} 行历史日期失败。值={val_b}, 文本={text_b}, 错误详情={e}")
            break
            
    if not last_date:
        print("未能在A列找到内容为 '平复' 的历史记录行，程序无法确定接着从哪天开始填写。")
        return
        
    print(f"查找到最后一组历史记录对应的日期为: {last_date.month}.{last_date.day}")
    
    # 起始日期为找到的日期的“明天”；终止日期为“今天”
    current_date = last_date + timedelta(days=1)
    today = datetime.now()
    
    if current_date.date() > today.date():
        print("文件中的日期已经是最新内容，不需要新增！")
        return
        
    # 一天一天遍历补充
    while current_date.date() <= today.date():
        # 组装格式（无前导零的 月.日）
        date_str = f"{current_date.month}.{current_date.day}"
        
        # 第1组内容：减一点
        ws.Cells(next_row, 1).Value = "减一点"
        ws.Cells(next_row, 2).Value = date_str
        ws.Cells(next_row, 3).Value = date_str
        next_row += 1
        
        # 第2组内容：平复
        ws.Cells(next_row, 1).Value = "平复"
        ws.Cells(next_row, 2).Value = date_str
        ws.Cells(next_row, 3).Value = date_str
        next_row += 1

        # 第3组内容：视觉学习
        ws.Cells(next_row, 1).Value = "视觉学习"
        ws.Cells(next_row, 2).Value = date_str
        ws.Cells(next_row, 3).Value = date_str
        next_row += 1
        
        print(f"写入成功: {date_str} 的 '减一点'、'平复' 和 '视觉学习'")
        current_date += timedelta(days=1)
        
    # 这里不要调用 _wb.Close()_ 或者 _excel.Quit()_
    print("今日数据更新全部完毕，Excel 将继续保持打开状态。")

if __name__ == "__main__":
    main()
