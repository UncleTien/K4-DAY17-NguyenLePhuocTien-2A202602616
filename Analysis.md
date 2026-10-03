# Day17 — Memory Systems for Agent: Benchmark Analysis

## 1. Kết quả benchmark

### Standard Benchmark

| Metric | Baseline | Advanced |
|---|---:|---:|
| Agent tokens only | 1494 | 1542 |
| Prompt tokens processed | 14187 | 19913 |
| Cross-session recall | 0.00 | 0.86 |
| Response quality | 1.00 | 1.00 |
| Memory growth (bytes) | 0 | 252 |
| Compactions | 0 | 0 |

### Long-Context Stress Benchmark

| Metric | Baseline | Advanced |
|---|---:|---:|
| Agent tokens only | 247 | 301 |
| Prompt tokens processed | 22077 | 14650 |
| Cross-session recall | 0.00 | 1.00 |
| Response quality | 1.00 | 1.00 |
| Memory growth (bytes) | 0 | 204 |
| Compactions | 0 | 5 |

---

## 2. Vì sao Advanced có recall tốt hơn Baseline?

Ở Standard Benchmark, `Cross-session recall` của Baseline là **0.00**, trong khi Advanced đạt **0.86**. Ở Long-Context Stress Benchmark, Baseline tiếp tục đạt **0.00**, còn Advanced đạt **1.00**.

Nguyên nhân nằm ở cách hai agent quản lý memory.

Baseline chỉ giữ conversation history bên trong từng thread. Khi chuyển sang thread mới, lịch sử của thread trước không còn nằm trong context nên agent không thể recall các thông tin người dùng đã cung cấp.

Advanced có thêm persistent memory thông qua `User.md`. Trong `memory_store.py`, `extract_profile_updates()` trích xuất các stable facts như tên, nơi ở, nghề nghiệp, đồ uống yêu thích và response style. Các fact này được ghi xuống `User.md`. Khi người dùng mở thread khác, `AdvancedAgent` đọc lại profile và `_offline_response()` sử dụng các fact đã lưu để trả lời recall question.

Kết quả này cũng thể hiện ở `Memory growth`: Baseline luôn bằng **0 bytes**, trong khi Advanced tăng **252 bytes** ở Standard Benchmark và **204 bytes** trong stress benchmark.

Điều đó cho thấy khả năng cross-session recall không đến từ conversation history dài hơn mà từ persistent memory được lưu riêng cho người dùng.

Giới hạn của cơ chế này là chất lượng recall phụ thuộc vào chất lượng fact extraction. Nếu extractor không nhận ra một fact, hoặc lưu một fact không chính xác, persistent memory có thể tiếp tục mang lỗi đó sang các session sau.

---

## 3. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Standard Benchmark cho thấy:

- Baseline `Agent tokens only`: **1494**
- Advanced `Agent tokens only`: **1542**
- Baseline `Prompt tokens processed`: **14187**
- Advanced `Prompt tokens processed`: **19913**

Như vậy trong benchmark ngắn, Advanced xử lý nhiều hơn **5726 prompt tokens**, tương đương khoảng **40.4%** so với Baseline.

Nguyên nhân là Advanced không chỉ xử lý recent conversation. Nó còn phải quản lý persistent profile và đưa các thông tin memory cần thiết vào context.

Trong khi đó, Standard Benchmark chưa đủ dài để vượt `compact_threshold_tokens`, vì vậy số lần `Compactions` của Advanced vẫn bằng **0**.

Nói cách khác, Advanced đã phải trả chi phí của memory nhưng chưa nhận được lợi ích của compaction.

Đây là một trade-off quan trọng: persistent memory cải thiện recall nhưng không miễn phí. Với conversation ngắn, một Baseline đơn giản có thể xử lý ít context hơn.

---

## 4. Vì sao compact có lợi thế ở hội thoại dài?

Long-Context Stress Benchmark cho kết quả:

- Baseline `Prompt tokens processed`: **22077**
- Advanced `Prompt tokens processed`: **14650**
- Baseline `Compactions`: **0**
- Advanced `Compactions`: **5**

Advanced xử lý ít hơn **7427 prompt tokens**, tương đương giảm khoảng **33.6%** so với Baseline.

Đây là cột quan trọng để đánh giá compact memory: **`Prompt tokens processed`**, không phải `Agent tokens only`.

`Agent tokens only` của Advanced thực tế vẫn cao hơn:

- Baseline: **247**
- Advanced: **301**

Điều này cho thấy compact không nhất thiết làm model sinh ít output token hơn. Thứ compact tối ưu là lượng historical context phải được đưa lại vào prompt qua nhiều lượt.

`CompactMemoryManager` giữ lại một số recent messages theo `compact_keep_messages`, trong khi các message cũ được chuyển thành summary. Khi context vượt `compact_threshold_tokens`, quá trình này được kích hoạt.

Stress benchmark đã kích hoạt **5 compactions**. Vì vậy Advanced không phải tiếp tục mang toàn bộ raw conversation history như Baseline.

Kết quả cho thấy lợi ích của compact memory chỉ trở nên rõ khi conversation đủ dài. Ở Standard Benchmark, Advanced tốn prompt hơn Baseline; ở Long-Context Benchmark, quan hệ này đảo chiều.

---

## 5. Memory file tăng trưởng và rủi ro

Persistent memory cũng tạo ra một chi phí khác ngoài token: storage tăng theo thời gian.

Trong benchmark:

- Standard Advanced memory growth: **252 bytes**
- Long-context Advanced memory growth: **204 bytes**
- Baseline: **0 bytes** trong cả hai trường hợp.

Các con số hiện tại còn nhỏ vì dataset nhỏ và `User.md` chỉ giữ một số stable facts. Tuy nhiên, nếu hệ thống liên tục ghi thêm preference, entity và profile fact mà không có cơ chế kiểm soát, file memory có thể tăng dần theo thời gian.

Rủi ro nghiêm trọng hơn kích thước file là **incorrect persistent memory**.

Ví dụ, benchmark có các correction như:

- nơi ở cũ → nơi ở mới;
- `backend engineer` → `MLOps engineer`;
- stress test thay đổi nơi ở từ `Huế` → `Đà Nẵng`.

Nếu hệ thống giữ cả fact cũ và fact mới mà không xử lý conflict, agent có thể recall thông tin lỗi ở những session sau.

Persistent memory vì vậy cần guardrail tốt hơn conversation memory thông thường: một lỗi được ghi vào persistent store có thể tồn tại lâu hơn thread đã tạo ra lỗi đó.

---

# 6. Trade-off tổng thể

Benchmark thể hiện một trade-off rõ ràng giữa simplicity, recall và context cost.

**Baseline** có thiết kế đơn giản và không phát sinh persistent memory. Trong conversation ngắn, nó cũng xử lý ít prompt token hơn. Tuy nhiên, khi chuyển session, recall giảm xuống **0.00**.

**Advanced** thêm persistent `User.md`, fact extraction và compact memory. Điều này làm Standard Benchmark tốn nhiều prompt hơn, nhưng cross-session recall tăng từ **0.00 lên 0.86**.

Khi conversation trở nên dài, compact memory bắt đầu phát huy tác dụng. Advanced giảm `Prompt tokens processed` từ **22077 xuống 14650**, đồng thời vẫn đạt cross-session recall **1.00**.

Do đó, Advanced không đơn giản là "tốt hơn" Baseline trong mọi trường hợp. Nó đánh đổi implementation complexity và memory-management overhead để đạt persistent recall và kiểm soát context growth tốt hơn trong hội thoại dài.

---

# 7. Bonus được chọn — Conflict Handling

## 7.1 Vấn đề cần giải quyết

Bonus được chọn là **Conflict Handling cho persistent user facts**.

Đây là vấn đề xuất hiện trực tiếp trong benchmark thay vì một feature được thêm tùy ý.

Dataset chứa nhiều correction, ví dụ:

- nơi ở thay đổi;
- nghề nghiệp thay đổi từ `backend engineer` sang `MLOps engineer`;
- stress test thay đổi location từ `Huế` sang `Đà Nẵng`;
- các thông tin gây nhiễu như nhắc `product manager` hoặc `Hà Nội` nhưng không phải fact hiện tại.

Một memory system chỉ append fact mới có thể giữ đồng thời hai giá trị mâu thuẫn.

## 7.2 Cách cải thiện hệ thống

Với các field được xem là single-valued stable facts, ví dụ:

- `name`
- `location`
- `profession`
- `response_style`

khi extractor phát hiện một correction đáng tin cậy, hệ thống nên **replace giá trị cũ bằng giá trị mới**, thay vì append thêm một fact thứ hai.

Ví dụ:

```text
location: Đà Nẵng
```

sau correction:

```text
Giờ mình đang ở Huế.
```

memory trở thành:

```text
location: Huế
```

thay vì:

```text
location: Đà Nẵng
location: Huế
```

Cơ chế này cải thiện recall vì câu hỏi về thông tin **hiện tại** chỉ có một canonical value để đọc.

Nó cũng giúp giới hạn memory growth vì các correction không tạo ra các dòng fact trùng lặp vô hạn.

## 7.3 Ảnh hưởng đến recall và token cost

Conflict handling chủ yếu cải thiện **recall correctness**.

Nếu fact cũ và mới cùng tồn tại, model phải tự suy luận giá trị nào hiện tại. Sau nhiều session hoặc sau compaction, thông tin temporal này có thể bị mất.

Việc giữ một canonical current value giúp Advanced đưa ít thông tin mâu thuẫn hơn vào prompt và giảm khả năng trả lời bằng stale fact.

Nó cũng có thể giảm nhẹ token/storage growth vì hệ thống update existing fact thay vì append history liên tục.

Tuy nhiên mục tiêu chính của bonus này vẫn là **correctness**, không phải token optimization.

## 7.4 Rủi ro mới

Conflict handling tạo thêm một rủi ro: hệ thống có thể **overwrite một fact đúng bằng một fact sai**.

Ví dụ người dùng nói:

```text
Có lúc mình đùa rằng hay là chuyển sang product manager.
```

Nếu extractor chỉ nhìn thấy cụm `product manager` mà không hiểu đây là câu giả định hoặc câu đùa, nó có thể ghi đè:

```text
profession: MLOps engineer
```

thành:

```text
profession: product manager
```

Khi đó persistent memory sẽ biến một extraction error thành lỗi tồn tại qua nhiều session.

Vì vậy conflict handling nên kết hợp với guardrail như:

- chỉ update khi câu chứa tín hiệu assertion/correction rõ ràng;
- không ghi fact từ câu hỏi;
- không coi hypothetical hoặc joke là correction;
- có thể thêm confidence threshold trước khi overwrite một fact quan trọng.

Điều này cho thấy bonus cải thiện memory correctness nhưng đồng thời làm hệ thống phức tạp hơn và tạo thêm yêu cầu về extraction quality.

---

# 8. Kết luận

Kết quả benchmark thể hiện đầy đủ chuỗi trade-off của memory system:

1. Baseline không có persistent memory nên cross-session recall bằng **0.00**.
2. Advanced dùng `User.md`, giúp Standard recall tăng lên **0.86** và stress recall đạt **1.00**.
3. Persistent memory tạo overhead trong conversation ngắn: Standard prompt processing tăng từ **14187** lên **19913**.
4. Khi conversation dài, compact memory đảo ngược trade-off này: prompt processing giảm từ **22077** xuống **14650**, với **5 lần compaction**.
5. Persistent memory tạo thêm storage growth và nguy cơ stale/incorrect fact.
6. Conflict Handling giúp giải quyết correction nhưng cần guardrail để tránh overwrite fact đúng bằng extraction sai.

Vì vậy thiết kế Advanced đạt recall và long-context efficiency tốt hơn, nhưng đổi lại cần nhiều logic quản lý memory và validation hơn Baseline.