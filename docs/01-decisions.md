# 01 — Kararlar ve kapsam

Durum: **Faz 1–5 sözleşmeleri, sınırlı canlı provider yüzeyi ve 50 episode'luk bounded automation regression'ı uygulandı; iki gerçek kaynakta modelsiz pilot kanıtı vardır**

Bu belgedeki V1 kararları hedef sözleşmeyi tarif eder. Bugün çalışan yüzey; strict schema, kimlik/hash, snapshot/ingest, artifact/event, kaynak-biçimli adapterlar, canonicalizer, tool registry, explicit-kanıtlı source-evidence Pass 1, duplicate/conflict audit, insan-kapılı selection freeze, field-policy/host merge, pre-egress guard, canlı provider adapterları, pre-review canary, bounded automation ve review/release kapılarını içerir. NVIDIA When2Call ve Salesforce xLAM 60k üzerinde modelsiz pilotlar tamamlanmıştır; ayrıca doğrulanmış `translation-prompt-0.3.0` / Flash 6-worker regression'ında 50 conflict-free adayın 46'sı çevrilmiş, 547 leaf'in 541'i kabul edilmiş ve 40 satırlık HF review paketi üretilmiştir. v0.4 aday prompt aynı cohortta bunu geçemediği için production'a alınmamıştır; canlı egress yalnız `promotion_status=validated` prompt bundle ile mümkündür. Bu, insan adjudication, S400 kabulü veya release izni değildir. Ayrıntılı kanıt tablosu için [06 — Uygulama durumu](06-implementation-status.md) belgesine bakın.

## 1. Başarı tanımı

Bir release başarılıdır, eğer:

- tüm kayıtlar tek konuşma şemasına uyuyorsa,
- kaynak, lisans, split ve ham satır kimliği izlenebiliyorsa,
- teknik alanlar çeviri öncesi/sonrası değişmemişse,
- duplicate ve conflict kararları yeniden üretilebiliyorsa,
- kabul edilen Türkçe metinlerde critical veya major anlam hatası yoksa,
- Parquet export üretilmişse JSONL ve Parquet mantıksal olarak aynı kayıtları taşıyorsa,
- release manifesti tüm girdileri ve sürümleri hash ile sabitliyorsa.

## 2. V1 kapsamı

Her canonical satır, son mesajı hedef assistant kararı olan tek bir episode'dur. Konuşma prefix'i tek veya çok turlu olabilir. Hedef karar iki alanla ifade edilir:

| `decision.action` | Açıklama |
|---|---|
| `tool_call` | Assistant bir veya daha fazla geçerli tool çağrısı yapar. |
| `clarification` | Assistant, çağrı yapmadan gerekli bilgiyi ister. |
| `tool_unavailable` | Uygun tool bulunmadığını veya işlemin yapılamadığını belirtir. |
| `direct_answer` | Tool sonucu kullanmadan doğrudan cevap verir. |
| `final_answer` | Önceki source-backed tool sonuçlarına dayanan final cevaptır. |

`decision.call_shape`, yalnız `tool_call` için `single` veya `multi_same_turn` olur. `multi_same_turn`, çağrıların gerçek hayatta paralel veya bağımsız çalıştığını kanıtlamaz. Kanıt yoksa `execution_topology=unknown` kalır.

## 3. V1 dışında kalanlar

- Kaynakta olmayan tool result veya final answer üretmek.
- Chain-of-thought ya da sentetik reasoning üretmek.
- Similarity skoruna bakıp otomatik kayıt silmek.
- Tool adını tek başına tool kimliği kabul etmek.
- Statik sözlükle kör search/replace yapmak.
- Judge kararını insan etiketiyle kalibre etmeden otomatik gold kabulü vermek.
- Başlangıçta Airflow, Spark, lakeFS, Iceberg/Delta veya genel amaçlı agent framework kurmak.

## 3.1 Kaynakta cevap/tool sonucu yoksa

Bir tool-calling kaydında assistant'ın `tool_calls` üretmesi başlı başına geçerli hedef cevaptır. Kaynak tool execution sonucu veya sonuçtan türetilmiş final assistant metni sağlamıyorsa:

- `assistant.content=null`, `assistant.tool_calls=[...]` korunur,
- `trajectory_state=awaiting_tool` atanır,
- `role=tool` mesajı eklenmez,
- final assistant cevabı uydurulmaz,
- kayıt yalnız tool seçimi ve argument prediction eğitimi için kullanılır.

Bu episode “complete grounded conversation” olarak etiketlenemez. İleride fixture-backed execution veya sentetik tamamlama denenirse ayrı provenance, ayrı kalite/release sınıfı ve ayrı eval kapısı kullanır; kaynak-backed gold kayıtlarla sessizce karıştırılmaz.

## 4. Temel mimari kararlar

### 4.1 Tek mantıksal satır

Her satır bir konuşmadır. Canonical JSONL tek doğruluk kaynağıdır. Manifest, indeks, event, rapor ve review kararları JSON/JSONL tutulur. Parquet yalnız aynı satırların isteğe bağlı Hugging Face yayın türevidir.

### 4.2 Tek hedef episode yaklaşımı

Her canonical satır tek hedef assistant kararıyla biter. Davranış etiketi tüm kaynak diyaloğa kör biçimde değil bu hedef mesaja atanır. Clarification sonrası yeni kullanıcı cevabı ve tool call geliyorsa aynı kaynak diyaloğun iki bağlı episode'u üretilir; `parent_episode_id` ilişkilerini korur.

### 4.3 Kimlikler birbirine karıştırılmaz

- `snapshot_id`: bir veya daha fazla kaynak dosyanın immutable snapshot'ı.
- `source_occurrence_id`: snapshot içindeki fiziksel kayıt konumu.
- `source_native_id`: kaynak güvenilir bir ID sağlıyorsa ayrı alan.
- `episode_id`: hedef source episode'una bağlı ve çeviri boyunca değişmeyen kimlik.
- `source_episode_fingerprint`: provenance'dan bağımsız İngilizce canonical episode içeriği; cross-source exact duplicate kimliği.
- `variant_id`: İngilizce, Türkçe veya repair varyantının içerik hash'i.
- `release_id`: canonical satırda değil, yalnız release membership manifestinde.

Bu ayrım sayesinde içerik değişimi, cross-source alias, çeviri varyantı ve release üyeliği birbirine karışmaz.

### 4.4 Tool kimliği ad + semantik schema'dır

Aynı tool adı farklı parameter şemalarıyla gelirse bunlar otomatik birleştirilmez. Tool identity şu girdilerden oluşur:

```text
tool_id = sha256(JCS({name, normalized_validation_schema}))
```

`description`, `title`, `$comment` ve `examples` yapısal kimlikten ayrıdır; dokümantasyon hash'ine girer. `default`, `required`, property adları ve enumlar davranışı etkileyebildiği için kimliğe girer.

### 4.5 Immutable stage çıktıları

Her aşama yeni artifact ve deterministik content manifesti üretir. Zaman/run ID içeren operasyon event'i bundan ayrıdır. Yerinde düzeltme veya overwrite yoktur. Hatalı kayıtlar gerekçesiyle `quarantine`a gider; sessizce düzeltilmez.

### 4.6 Pilot sürümleme politikası

İlk çalışan sözleşmeler `0.1.0` ile başlar. Pilot sırasında schema kararlı kabul edilmez:

```text
0.1.0   ilk implementasyon ve fixture testleri
0.2.x   S30 sonrası uyumlu düzeltmeler
0.x     S100/S250/S400 pilot sözleşmeleri
1.0.0   S400 gold, migration ve uyumsuz sorunlar tamamlandı
```

`canonical_schema`, `tool_registry`, `normalizer`, `source_evidence`, `diagnostic_catalog`, `translation_memory`, `render_schema` ve `release_manifest` ayrı sürümlenir. Bir sürümün anlamı sonradan değiştirilmez. Geriye uyumsuz değişiklik yeni sürüm ve açık migration ister; eski JSONL sessizce yeniden yorumlanmaz.

## 5. Teknoloji kararları

| Alan | Seçim | Gerekçe |
|---|---|---|
| Runtime | Python 3.12+ | Ekosistem ve veri araçları güçlü. |
| Paket yönetimi | uv | Hızlı, platformlar arası, kilitli ortam. |
| Veri modeli | Pydantic v2 strict | Runtime validation ve schema üretimi aynı kaynaktan. |
| Schema | JSON Schema 2020-12 | Tool ve kayıt sözleşmesi için güncel standart. |
| Canonical JSON | RFC 8785/JCS | Tekrarlanabilir hash ve kimlik. |
| CLI | Typer + Rich | Basit alt komutlar ve okunabilir inceleme. |
| Çalışma verisi | JSON + JSONL | İnsan tarafından doğrudan okunabilir ve diff edilebilir. |
| İndeks/olaylar | Sıralı ve sharded JSONL | Veritabanı olmadan streaming anti-join ve audit. |
| İsteğe bağlı HF export | PyArrow | Yalnız release anında açık şemalı Parquet üretmek için. |
| Büyük dönüşüm | Polars yalnız ölçümle gerekirse | V1'e gereksiz bağımlılık eklememek için başlangıçta zorunlu değil. |
| Test | pytest + Hypothesis | Örnek tabanlı ve property-based invariant testleri. |
| Kalite | DeepSeek + GPT katmanları + insan | Maliyet, hız ve kaliteyi ayrı sorumluluklarla dengelemek. |

SQLite, DuckDB ve başka bir veritabanı v1 kapsamı dışındadır. Performans sorunu varsayılmayacak; ancak ölçülürse JSONL shard/index tasarımı önce iyileştirilecektir.

## 6. Model rolleri

Model adları kod içine dağılmaz; sürümlü run config'inde tutulur. `2026-08-13`
itibarıyla config allow-list'i resmî provider kataloglarındaki model kimlikleriyle
eşleşir; bu, her hesabın kota/erişim yetkisini garanti etmez.

- Çevirici: [`deepseek-v4-flash`](https://api-docs.deepseek.com/news/news260424/) ; başlangıçta thinking kapalı, sıcaklık 0. `deepseek-v4-pro` yalnız allow-list'li alternatifidir.
- Güçlü semantik yargıç: [`gpt-5.4`](https://developers.openai.com/api/docs/models/gpt-5.4).
- Dar adversarial doğrulayıcı: [`gpt-5.4-mini`](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
- Terminoloji araştırmacısı: mini model + web search; çelişki halinde güçlü modele veya insana yükseltme.
- Challenger: daha yeni model ancak frozen eval setinde üstünlük kanıtlandıktan sonra terfi eder.

Her gerçek çağrıda endpoint/provider, istenen model adı, çözümlenen snapshot/system fingerprint, reasoning ayarı, decode/token parametreleri, SDK sürümü, prompt/schema/config hash'i ve ham cevap hash'i kaydedilir. Provider fingerprint değişikliği canary ve 30-kayıt kapısını yeniden açar.

## 7. Kaynak doğruluğu, gold ve silver ayrımı

İyi Türkçe, yanlış kaynak davranışını düzeltmez. Çeviriden önce source episode ayrıca semantik kapıdan geçer:

- `source_valid`: seçim havuzuna girebilir.
- `source_review`: insan kararı bekler.
- `source_invalid`: çeviri havuzuna giremez; kaynak kanıtıyla karantinada kalır.

Model source kaydı otomatik düzeltmez veya yeniden etiketlemez.

Bu kapı iki geçişlidir: bütün kaynaklarda ucuz deterministik kanıt çıkarılır; pahalı semantik inceleme yalnız structurally eligible ve seçim sırasındaki adaylara uygulanır. S400, 400 adet insan-adjudicated `source_valid` episode bulunana kadar ranked reserve queue'dan doldurulur. Böylece 58 bin kaydın tamamına model çağrısı yapılmaz.

## 7.1 Bağlamlı tam-segment translation memory

Memory bir sözlük veya model cache'i değildir. Yalnız şu koşullar birlikte exact eşleşirse insan-kabul edilmiş tam hedef segment yeniden kullanılabilir:

- kaynak segmentin exact UTF-8 hash'i,
- field/path sınıfı ve argument policy,
- source/presented context fingerprint'i,
- tool/documentation kapsamı,
- decision/action kapsamı,
- hedef locale ve policy sürümleri.

Fuzzy, kelime veya alt-parça eşleşmesi reuse yapmaz. Aynı exact key için farklı kabul edilmiş hedefler varsa otomatik seçim yerine `memory_conflict` oluşur. Prompt/eval karşılaştırmalarında memory kapalıdır; production reuse bütün normal kalite kapılarından geçer ve otomatik gold hakkı vermez.

## 7.2 Training render ve loss mask

Canonical JSONL modelden bağımsız kalır. Her hedef model ailesi için sürümlü bir training view üretilir. V1 label politikası yalnız satırın sonundaki hedef assistant mesajını supervise eder; önceki system/developer/user/tool ve assistant prefix tokenları maskelenir. xLAM tool-call-only episode'unda hedef assistant tool-call serialization'ı label'dır.

Target veya gerekli tool tanımı truncation ile kesilirse kayıt sessizce kısaltılmaz; `render_ineligible` olur. Render geçişi canonical veriyi değiştirmez ve her model/template/tokenizer için ayrı manifest üretir.

- `silver`: deterministik kontrolleri ve model kalite kapılarını geçmiş, ilgili satırların tamamı insan tarafından kabul edilmemiş veri.
- `gold`: her satırı deterministik kontrollerden geçmiş, model değerlendirmeleri kaydedilmiş ve insan tarafından tek tek kabul edilmiş veri.

İlk 400 kayıt için insan incelemesi %100'dür. Daha büyük ölçekte örneklemeli insan incelemesi kullanılırsa yalnız insanın tek tek kabul ettiği alt küme `gold`, kalan uygun kayıtlar `silver` olur; otomatik kabul hiçbir zaman gold etiketi alamaz.

## 8. Başlıca standartlar ve kaynaklar

- [RFC 8785 JSON Canonicalization Scheme](https://www.rfc-editor.org/info/rfc8785/)
- [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12)
- [Hugging Face TRL konuşma ve tool-calling formatları](https://huggingface.co/docs/trl/en/dataset_formats)
- [Hugging Face Datasets JSON feature](https://huggingface.co/docs/datasets/about_dataset_features)
- [MQM çeviri kalite çerçevesi](https://www.jostrans.org/article/view/8074)
- [W3C PROV-DM — Provenance Data Model](https://www.w3.org/TR/prov-dm/)
- [C2PA Technical Specification](https://c2pa.org/specifications/)
- [MLCommons Croissant — ML-ready dataset metadata](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html)
- [RO-Crate — research object packaging](https://www.researchobject.org/ro-crate/)
- [Datasheets for Datasets (Gebru et al.)](https://arxiv.org/abs/1803.09010)
- [Model Cards (Mitchell et al.)](https://arxiv.org/abs/1810.03993)
- [DVC — Data Version Control](https://dvc.org/doc)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [EU AI Act, Article 50 — AI-content transparency](https://artificialintelligenceact.eu/article/50/)
