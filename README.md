# Project Parallel Computing with CUDA

สมาชิกกลุ่ม
1. นายณัฐวรรธน์ สุนทรเสถียรกุล
2. นายพันธุ์ธิชชัย ขรรค์บริวาร 

## Sequential Dijkstra Algorithm
![image](https://user-images.githubusercontent.com/48822642/115367429-64c83f00-a1f0-11eb-9755-cfdd421b3865.png)

## Parallel Dijkstra Algorithm
![image](https://user-images.githubusercontent.com/48822642/115367545-80cbe080-a1f0-11eb-88a0-5f82e870029c.png)

## Result
จากผลการทดสอบการทำงานของอัลกอริธึมทั้งสองรูปแบบ ได้ผลออกมาว่า การทำงาน Dijkstra Algorithm แบบ Sequential ใช้เวลาทำงานทั้งหมด 6 milliseconds และการทำงาน Dijkstra Algorithm แบบ Parallel ใช้เวลาทั้งหมด 0.037888 milliseconds ทำให้สรุปได้ว่า การทำงานแบบ Parallel นั้นทำงานได้เร็วกว่าแบบ Sequential อยู่หลายเท่าตัว(ในที่นี้คือ 158.4 เท่า เมื่อเทียบจากความเร็วในการทำงาน)
