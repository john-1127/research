#!/bin/bash

# 抓16 80  直到90~100
# 想要matching species were found.
# wget -O ./fine-tune/result.txt "https://webbook.nist.gov/cgi/cbook.cgi?Value=16+80&VType=MW&Formula=C&AllowOther=on&AllowExtra=on&Units=SI&cIR=on"
# python ./fine-tune/beauty.py

# 要分段 80 120(+5), 120 210(+3), 210 500(+5),
# for ((low = 210; low < 500; low += 5)); do
#   high=$((low + 5))
#   echo "Searching ${low}~${high}..." >>./fine-tune/log.txt
#
#   # 下載結果
#   wget -O ./fine-tune/result.txt "https://webbook.nist.gov/cgi/cbook.cgi?Value=${low}+${high}&VType=MW&Formula=C&AllowOther=on&AllowExtra=on&Units=SI&cIR=on"
#
#   # 檢查是否有找到分子
#   if grep -q "matching species were found." ./fine-tune/result.txt; then
#     echo "${low}~${high} found." >>./fine-tune/log.txt
#     python ./fine-tune/beauty.py
#
#   else
#     echo "${low}~${high} not found, skip" >>./fine-tune/log.txt
#   fi
done
