# Project Parallel Computing with CUDA

สมาชิกกลุ่ม
1. นายณัฐวรรธน์ สุนทรเสถียรกุล
2. นายพันธุ์ธิชชัย ขรรค์บริวาร 

## Floyd Warshall Algorithm
![image](https://user-images.githubusercontent.com/48822642/117423584-1387b180-af4b-11eb-94b5-dc3daac657f8.png)


## Result
จากผลการทดสอบการทำงานของอัลกอริธึมทั้งสองรูปแบบ ได้ผลออกมาว่า เมื่อมีจำนวนโหนดเป็น 4096 โหนด 

การทำงาน Floyd Warshall Algorithm แบบ Sequential ใช้เวลาทำงานทั้งหมด 188011.99341 milliseconds และ

การทำงาน Floyd Warshall Algorithm แบบ Parallel ใช้เวลาทั้งหมด 5402.25293 milliseconds 

ทำให้สรุปได้ว่า การทำงานแบบ Parallel นั้นทำงานได้เร็วกว่าแบบ Sequential อยู่หลายเท่าตัว(ในที่นี้คือ 34.80252 เท่า เมื่อเทียบจากความเร็วในการทำงาน)
