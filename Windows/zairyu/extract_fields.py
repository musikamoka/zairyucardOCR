import re

def extract_fields(text):
    return {
        "国籍": re.search(r"国籍[:：]?\s*([^\s\n]+)", text).group(1) if re.search(r"国籍[:：]?\s*([^\s\n]+)", text) else "",
        "氏名": re.search(r"氏名[:：]?\s*(.+)", text).group(1).strip() if re.search(r"氏名[:：]?\s*(.+)", text) else "",
        "性別": re.search(r"性別[:：]?\s*(男|女)", text).group(1) if re.search(r"性別[:：]?\s*(男|女)", text) else "",
        "生年月日": re.search(r"生年月日[:：]?\s*(\d{4}年\d{1,2}月\d{1,2}日)", text).group(1) if re.search(r"生年月日[:：]?\s*(\d{4}年\d{1,2}月\d{1,2}日)", text) else "",
        "住居地": extract_multiline_address(text),
        "在留資格": re.search(r"在留資格[:：]?\s*(\S+)", text).group(1) if re.search(r"在留資格[:：]?\s*(\S+)", text) else "",
        "在留カード番号": extract_card_number(text),
        "在留期間_期間": extract_stay_period(text)[0],
        "在留期間_満了日": extract_stay_period(text)[1],
    }

def extract_multiline_address(text):
    match = re.search(r"住居地[:：]?\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return ""

def extract_stay_period(text):
    match = re.search(
        r"在留期間(?:\(PERIOD OF STAY\))?.*?[:：]?\s*(\d+年).*?\(?(\d{4}年\d{1,2}月\d{1,2}日)",
        text
    )
    if match:
        return match.group(1), match.group(2)  # 正确顺序：期間, 満了日
    return "", ""

def extract_card_number(text):
    match = re.search(r"右上の番号は在留カード番号です[:：]?\s*([A-Z0-9]+)", text)
    if match:
        return match.group(1)
    return ""