# zairyucardOCR
このコードは、現実世界で日本の在留カードの写真撮影後、画像内の情報から在留カードの各種情報を抽出するものです。これにより、銀行、クレジットカード、学校などへの情報入力が容易になり、
手動入力の必要なく、オフラインで直接CSVファイルに変換できます。推奨構成：GPU:A100、RAM:12GB以上、CPU:8コア8スレッド<br>
そしたら　おすすめ環境はGoogle Colab PROプラン<br>
＞＞＞＞＞Google Colab PRO＜＜＜＜<br>
ファイルは　”zairyucardOCRbyDAIICHI23TE465ZHA.ipynb　”　ご覧ください<br>
＞＞＞＞＞Windows＜＜＜＜<br>
ファイルは　”zairyu”　ご覧ください<br>
[--app.py<<<こちらはMain　ソースコードです<br>
[--extract_fields.py<br>
[--model_runner.py<br>
必要環境はDeepSeeK Vl2です<br>
＊＊＊但しこのコードはCPU使えだけです、制限tokenは３００。自分のcpuはi9 12900Kfなので、写真解析は３０秒が必要（16C24T）＊＊＊<br>
＊＊＊＊＊＊こちらMITと日本知的財産の法律を応用されていますので、著者を明記せずに使用する場合、または違法な目的で使用する場合はご注意ください。＊＊＊<br>
＊＊＊＊＊＊This is based on the laws of MIT and Japanese intellectual property. Please be careful if you use it without specifying the author, or if you use it for illegal purposes.＊＊＊<br>
＊＊＊＊＊＊本内容适用于MIT及日本知识产权法，请注意，如在未注明作者的情况下使用，或用于非法目的，需谨慎。＊＊＊<br>
![image](https://github.com/user-attachments/assets/56fa4992-2229-4b7a-a489-b9a5cc0e67a7)
![QQ_1750697498283](https://github.com/user-attachments/assets/abc04888-baba-4d46-b5d0-0264b5289811)

![image](https://github.com/user-attachments/assets/39034d26-3b9b-41a6-83f0-0605c3ec302a)

