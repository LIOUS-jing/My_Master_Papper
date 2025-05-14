#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清洗Excel文件文本数据工具

此脚本用于清洗Excel文件中的文本数据，去除URL、HTML标签和特殊字符，
同时保留中文字符、标点符号和表情符号(emoji)。

用法:
    python xlsx_text_cleaner.py <输入文件.xlsx> <输出文件.xlsx> [配置]
    
配置格式:
    "工作表名:列名,工作表名:列名,..."
    例如: "{小红书final}:内容,{微博final}:博文内容"
"""

import os
import sys
import re
import pandas as pd
import regex

def clean_text(text):
    """
    清洗文本数据，移除URL、HTML标签和特殊字符，保留中文字符、标点符号和表情符号(emoji)。
    
    参数:
        text (str): 需要清洗的文本
        
    返回:
        str: 清洗后的文本
    """
    # 检查是否为NaN或非字符串类型
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 保留中文字符、英文字母、数字、表情符号和常用标点符号，移除其他特殊字符
    # 中文字符范围：[\u4e00-\u9fff]
    # 中文标点符号：[\u3000-\u303f\uff00-\uffef]
    # 表情符号范围：[\p{Emoji}]
    # 英文字母、数字和常用标点符号：[a-zA-Z0-9.,!?;:'"()[\]{}，。！？；：''""（）【】「」『』]
    clean_pattern = r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\p{Emoji} a-zA-Z0-9.,!?;:\'"()\[\]{}\s，。！？；：''""（）【】「」『』]'
    text = regex.sub(clean_pattern, '', text)
    
    # 标准化空白字符（将多个空格合并为一个）
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text

def clean_excel_multi_sheets(input_file, output_file, sheet_column_config):
    """
    清洗Excel文件中多个工作表的指定列的文本数据。
    
    参数:
        input_file (str): 输入Excel文件路径
        output_file (str): 输出Excel文件路径
        sheet_column_config (list): 包含工作表名和列名的配置列表，格式为[(sheet_name, column_name), ...]
        
    返回:
        str: 输出Excel文件路径
    """
    try:
        # 创建ExcelWriter对象以保存到同一个Excel文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 处理每个工作表和列
            for sheet_name, column_name in sheet_column_config:
                print(f"处理工作表 '{sheet_name}' 的 '{column_name}' 列...")
                
                # 读取工作表
                try:
                    df = pd.read_excel(input_file, sheet_name=sheet_name)
                except ValueError:
                    print(f"错误：工作表 '{sheet_name}' 在Excel文件中不存在")
                    continue
                
                # 检查列名是否存在
                if column_name not in df.columns:
                    print(f"错误：列 '{column_name}' 在工作表 '{sheet_name}' 中不存在")
                    print(f"可用的列名：{', '.join(df.columns)}")
                    continue
                
                # 对指定列应用清洗函数
                print(f"正在清洗 '{sheet_name}.{column_name}' 中的文本...")
                df[column_name] = df[column_name].apply(clean_text)
                
                # 保存清洗后的数据到指定工作表
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
        print(f"已完成清洗，结果保存至: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"处理出错: {str(e)}")
        sys.exit(1)

def parse_config(config_str):
    """
    解析配置字符串，获取工作表和列名配置
    
    参数:
        config_str (str): 格式为 "sheet1:col1,sheet2:col2,..."
        
    返回:
        list: [(sheet1, col1), (sheet2, col2), ...]
    """
    result = []
    items = config_str.split(',')
    for item in items:
        if ':' in item:
            sheet, column = item.split(':', 1)
            result.append((sheet.strip(), column.strip()))
        else:
            print(f"警告: 配置项 '{item}' 格式不正确，将被忽略")
    return result

def main():
    # 检查命令行参数
    if len(sys.argv) < 3:
        print("用法: python xlsx_text_cleaner.py <输入文件.xlsx> <输出文件.xlsx> [配置]")
        print("配置格式: \"工作表名:列名,工作表名:列名,...\"")
        print("例如: \"{小红书final}:内容,{微博final}:博文内容\"")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 默认配置
    default_config = "{小红书final}:内容,{微博final}:博文内容"
    config_str = sys.argv[3] if len(sys.argv) > 3 else default_config
    
    # 解析配置
    config = parse_config(config_str)
    if not config:
        print("错误: 无效的配置")
        sys.exit(1)
    
    print(f"将处理以下工作表和列:")
    for sheet, col in config:
        print(f"- 工作表 '{sheet}' 的 '{col}' 列")
    
    clean_excel_multi_sheets(input_file, output_file, config)

if __name__ == "__main__":
    main() 