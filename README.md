# Toolcall TR Pipeline

Bu proje, İngilizce tool-calling ve no-tool konuşma verilerini yüksek kaliteli Türkçe eğitim verisine dönüştürmek için tasarlanmış bağımsız bir veri hattıdır.

Kod deposu: [BilalAbic/toolcall-tr-pipeline](https://github.com/BilalAbic/toolcall-tr-pipeline).

Şu anki durum: **Faz 1–5 veri hattı ile fail-closed canlı çeviri ve judge operasyon yüzeyleri uygulanmıştır**. ID-first ingest, strict sözleşmeler, diagnostic catalog, JSONL artifact/event altyapısı, canonicalizer, tool registry, source-evidence, audit ve insan-kapılı S400 selection freeze çalışır durumdadır. `pilot run`, kaynağı değiştirmeden snapshot → ingest → canonical → audit zincirini işletir; reviewer JSONL kararları append-only zincire eklenir, release yalnız açık insan Gold kabulüyle oluşur. `translate` yalnız açık canonical JSONL'nin policy-izinli leaf'lerini işler; `evaluation run` yalnız tam-leaf hash'li girdileri judge eder ve Gold üretemez. Canlı katman; DeepSeek V4 Chat Completions çeviricisini ve OpenAI Responses structured-output judge'ını ayrı adapterlarda, allow-list'li `.env` resolver, HTTPS/no-redirect transport, secret/PII/local-path preflight ve hash-only attempt provenance ile bağlar. Sabit sentetik smoke'lar ile 3 episode'luk çeviri batch'i ve 1 girdilik judge batch'i canlıda geçti. NVIDIA When2Call ve Salesforce xLAM 60k gerçek kaynakları revision-pinned, modelsiz pilotlardan geçti; insan kararı, training render, gerçek kaynak model egress'i veya release üretilmedi.

Uygulanan yüzey ile planlanan sonraki fazların kesin ayrımı için [uygulama durumuna](docs/06-implementation-status.md) bakın.
Gerçek kaynak revision/hash/pilot sayımları için [kaynak pilot kayıtlarına](docs/07-source-pilots.md) bakın.

## Ürün hedefi

Tek mantıksal veri formatında, her satırı tek bir hedef assistant kararında biten tek veya çok turlu bir konuşma episode'u olan dataset üretmek:

- single-turn ve multi-turn konuşmalar,
- tek tool çağrısı,
- aynı assistant turundaki çoklu tool çağrıları,
- clarification,
- tool unavailable / no-tool davranışı,
- yalnız kaynakta gerçekten varsa tool result ve grounded final answer.

Çalışma ve denetim formatları JSON ile JSONL olacaktır. Hugging Face için gerekirse en son aşamada Parquet üretilir; Parquet hiçbir zaman veri kaynağı veya çalışma veritabanı değildir.

## Değişmez ilkeler

1. Her fiziksel kaynak kayda, parse veya dedup işleminden önce snapshot + dosya yolu + byte offset tabanlı kalıcı bir `source_occurrence_id` ve snapshot-içi sıra numarası verilir.
2. Kaynak satırlar yerinde değiştirilmez; her aşama yeni ve hash'lenmiş bir artifact üretir.
3. Yakın benzerlik hiçbir kaydı otomatik silmez. Yalnız inceleme adayı üretir.
4. Aynı bağlamda farklı davranış görüldüğünde kayıtlar silinmez; conflict kuyruğuna alınır.
5. Model yalnız çevrilebilir doğal dil parçalarını üretir. Tool adı, parametre anahtarı, enum, sayı tipi, rol, ID ve çağrı yapısı model tarafından yeniden yazılmaz.
6. Tool sonucu veya final cevap kaynakta yoksa uydurulmaz.
7. Araştırma, yalnız belirsiz terminoloji için ayrı ve sınırlı bir ajan tarafından yapılır; çevirici serbestçe web'e çıkmaz.
8. İnsan tarafından kabul edilmemiş çıktı `gold` olarak yayımlanmaz.
9. Aynı kaynak occurrence'ı, ham kayıt veya canonical episode JSONL olay günlüğüne düştükten sonra yeni kayıt diye tekrar seçilmez; yeni prompt denemesi ise açıkça yeni bir attempt olarak izlenebilir.
10. Prompt, model, schema, kod ve veri snapshot sürümleri her attempt ile kaydedilir.
11. Kaynak davranışı doğrulanmadan çeviri yapılmaz; iyi çeviri yanlış tool kararını meşrulaştıramaz.
12. Translation memory yalnız insan kabul edilmiş tam segment + exact bağlam eşleşmesidir; kelime/fuzzy ikame yapmaz.
13. Canonical kayıt, hedef modelin chat template ve loss mask testi geçmeden training-ready sayılmaz.

## Mevcut uygulama yüzeyi

- JSONL kaynak snapshot kaydı, dosya hash'i ve satır sayımı,
- parse işleminden önce byte-offset tabanlı occurrence kimliği,
- duplicate key, non-finite sayı, UTF-8/Unicode, boş/bozuk ve fazla büyük kayıtlar için fail-closed quarantine,
- strict/frozen Pydantic modeller ve Draft 2020-12 JSON Schema artifactları,
- RFC 8785/JCS tabanlı hash ve kalıcı kimlikler,
- içerik-adresli, overwrite etmeyen artifact yayını ve deterministik manifest,
- doğrulanan append-only event hash zinciri ve sabit satır sayılı JSONL shardları,
- xLAM fixture, Salesforce xLAM 60k embedded-JSON/APIGen ve NVIDIA When2Call adapterları; JSON-dizi kaynaktan immutable JSONL türetme,
- JSON Schema semantiğini koruyan tool normalizasyonu ve ayrı structural/documentation hash'leri,
- canonical role/call state machine, context/behavior fingerprintleri ve yeniden üretilebilir membership indexi,
- explicit JSON Pointer kanıtından deterministik source-evidence Pass 1; `unknown` ve `must_not_infer` fail-closed,
- exact alias/conflict audit, review-only n-gram/Jaccard near-duplicate candidate retrieval, append-only human conflict adjudication logu ve split-leakage guard,
- deterministik reserve queue ile yalnız insan-adjudicated `source_valid` adaylardan S400 freeze; S30/S100/S250 strict prefix membershipleri,
- yalnız policy-kapsamlı doğal dil leaf'lerini çıkaran ve teknik alanları hostta immutable tutan field-policy/merge sözleşmesi,
- model istemcisi olmadan strict segment request/response, sentinel/NFC/coverage doğrulaması ve secret/PII/local-path/private-endpoint için her durumda bloklayan pre-egress guard,
- immutable altı-katmanlı prompt bundle ve kayıtlı JSON yanıtla test edilen, ağsız fake-provider sınırı,
- yalnız exact/full-context eşleşen human-promoted segment memory ve deterministic terminology-risk/research metadata; fuzzy lookup veya fetch yok,
- DeepSeek V4 Chat Completions çevirici ve OpenAI Responses judge adapterları: provider-bağımlı wire format, local strict response/sentinel/MQM doğrulaması, exact public endpoint allow-list'i, bounded HTTPS/no-redirect transport ve secret-safe credential resolver,
- canlı preflight: payloaddaki secret, PII ve local-path bulgularında transport öncesi ret; yalnız fixed synthetic smoke için explicit `--live --send` yürütmesi,
- explicit `pilot run`: kaynakla kesişmeyen output altında snapshot/ingest/canonical/audit; quarantine veya çözülmemiş conflict olduğunda fail-closed,
- reviewer-authored tek-kayıt JSONL kararını doğrulayan append-only review komutları; Gold release için model verdict + insan kabul bağlantısı zorunlu,
- hash-only provider attempt kayıtları: ham request/response/credential tutulmaz; otomatik retry bütçesi `0`, geçici hata yalnız manual-retry adayıdır,
- explicit canlı çeviri: policy-izinli leaf başına immutable claim/attempt/checkpoint, kaynak rehash ve host-only merge; otomatik retry, Gold ve release yok,
- explicit canlı judge: tam-leaf content hash'iyle bağlı strict source/target girdileri, immutable result/attempt manifestleri ve `gold_release_allowed=false`,
- yalnız enjekte edilmiş renderer ve tokenizer ile çalışan render/loss-mask sözleşmesi: pinli render config, tek final-assistant payload aralığı, teknik yapı hash'i ve truncation/uyuşmazlıkta fail-closed; chat-template veya tokenizer indirme yok,
- yalnız yerel sentetik JSONL için Gold release-manifest sözleşmesi: sıralı dosya byte-hash/row-count kimliği, episode sırası ve açık insan kabul kimliği doğrulaması; Parquet, Dataset Card, Hugging Face veya yayın işlemi yok,
- `source register`, `source validate`, `source json-array-to-jsonl`, `source evidence`, `ingest`, `registry build`, `canonicalize`, `audit duplicates`, `audit near-duplicates`, `select freeze`, `pilot run`, `translate`, `evaluation run`, `review submit-evaluation`, `review submit-conflict`, `release build`, `release validate`, `inspect`, `stats`, `events show` ve `diagnostics` CLI komutları.

Gerçek eval UI, render ve yayın komutları hâlâ kapalıdır. Review/release komutları yalnız dışarıdan sağlanan insan kararlarını doğrular; karar üretmez. Canlı çeviri/judge çağrıları yalnız explicit `--live`, non-default config, ayrı output kökü ve preflight ile gerçekleşir; kaynak dataset asla varsayılan olarak gönderilmez.

## Yerel doğrulama

Python 3.12 kurulu bir ortamda:

```powershell
uv sync --frozen
uv run python scripts/export_schemas.py
uv run pytest
uv run ruff check .
uv run pyright
```

Schema artifactları `schemas/0.1.0/` altında üretilir. Testler yalnız sentetik fixture ve geçici dizinleri kullanır; gerçek veri dosyalarını değiştirmez.

## Gerçek kaynak pilotu

Kaynak JSONL, revision ve lisans bilgisi hazır olduğunda ilk adım yalnız aşağıdaki modelsiz kapıdır. Çıktı yolu kaynak köküyle kesişemez; salt-okunur kaynak dosyası yeniden hash'lenir.

```powershell
uv run tcdata pilot run <source.jsonl> --output <disjoint-output> --dataset <namespace> --revision <revision> --license <license-id> --license-url <license-url> --source-config <config> --source-split <split> --adapter <source-adapter> --run-event-id <run-id>
```

Pilot geçmeden çeviri veya Gold release açılmaz. Human-review ve release komutları, gerçek model verdict JSONL ve reviewer tarafından yazılmış tek-kayıt karar JSONL'si gerektirir; bunlar olmadan fail-closed kalır.

2026-08-13'te [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call) revision `0582f7749df63a96fdc3070932e83e72396ace53` altındaki tüm erişilebilir veri split'leri salt-okunur pilotta işlendi. `train/sft` ve `train/pref` içindeki seçilmiş assistant hedefleri artık dahil edilir: açık `<TOOLCALL>` işaretleri, açık ek-bilgi istemleri ve açık tool-unavailable metinleri canonical'a alınır; belirsiz veya schema/argument açısından geçersiz kayıtlar karantinada kalır. 27.952 satırdan 27.393 canonical survivor üretildi; 559 gerekçeli karantina var. Dört split birlikte 2.917 exact duplicate group ve 136 human-review conflict adayı üretti. Bu modelsiz bir işlemdir; hiçbir gerçek kaynak satırı provider'a gönderilmedi, Gold/release veya insan kararı üretilmedi.

Kullanıcının Hugging Face gated erişimiyle [Salesforce xLAM 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) train kaynağı revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866` ile indirildi. 96.1 MB JSON dizi, tam dosya hash'i bağlı immutable JSONL'ye dönüştürüldü ve tamamı modelsiz pilotta işlendi: 60.000 source-valid satırdan 57.718 canonical kayıt, 2.282 gerekçeli karantina ve insan review bekleyen 6 hard-conflict adayı oluştu. Kaynak veya karantina kayıtları düzeltilmedi; gerçek kaynak içeriği provider'a gönderilmedi.

Pilot ve selection kapıları geçtikten sonra sınırlı bir batch için canlı adımlar şunlardır:

```powershell
uv run tcdata translate <canonical.jsonl> --output <disjoint-translation-output> --config configs/provider-smoke.toml --field-policy configs/field_policy.toml --prompt configs/prompt_bundle.toml --live
uv run tcdata evaluation run <live-evaluation-input.jsonl> --output <disjoint-evaluation-output> --config configs/provider-smoke.toml --role mini_verifier --run-id <run-id> --live
```

İki komut da otomatik Gold/release yapmaz ve ham içerik ya da anahtar tutmayan attempt manifestleri üretir.

## Belgeler

- [Kararlar ve kapsam](docs/01-decisions.md)
- [Veri sözleşmesi ve aşamalar](docs/02-data-pipeline.md)
- [Prompt, araştırma ve kalite](docs/03-prompt-research-eval.md)
- [Uygulama planı ve kabul kapıları](docs/04-implementation-plan.md)
- [Riskler ve üretime hazırlık](docs/05-risk-and-readiness.md)
- [Faz 1–5 uygulama durumu ve doğrulama](docs/06-implementation-status.md)
- [Gerçek kaynak pilot kayıtları](docs/07-source-pilots.md)

## Teknoloji yönü

- Python 3.12+
- `uv` ile bağımlılık ve lockfile yönetimi
- Pydantic v2 strict modeller + JSON Schema Draft 2020-12
- Typer + Rich CLI
- JSON/JSONL dosyaları üzerinde streaming inceleme, indeks ve raporlama
- Yalnız Hugging Face release gerekiyorsa PyArrow ile açık şemalı Parquet export
- RFC 8785/JCS + SHA-256 ile deterministik kimlikler
- pytest + Hypothesis + Ruff + Pyright
- DeepSeek V4 çeviri adapterı ve OpenAI Responses judge adapterı; kullanım yalnız explicit canlı kapı ve preflight ardında,
- OpenAI mini/strong judge rolleri için strict structured-output sözleşmesi; Gold kabulü yine insan yetkisinde,
- ihtiyaç halinde, sonraki fazda kaynak gösterecek terminoloji araştırma ajanı.

V1'de ağır orchestration framework'ü, otomatik sözlük ikamesi, near-duplicate otomatik silme, açık uçlu otonom ajan ve gereksiz data-lake altyapısı kullanılmayacaktır.

V1'de SQLite, DuckDB veya başka bir veritabanı kullanılmayacaktır. İnsan tarafından okunabilir JSON/JSONL artifact'ları tek doğruluk kaynağıdır.

## API anahtarları

`.env` yalnız `DEEPSEEK_API_KEY` veya `OPENAI_API_KEY` allow-list'i üzerinden explicit canlı komutta okunur; değerler loglara, artifactlere veya eventlere yazılmaz. Varsayılan `configs/pipeline.toml` çevrimdışıdır. Sağlayıcı bağlantısını güvenle doğrulamak için sabit sentetik smoke komutları kullanılır:

```powershell
uv run tcdata provider smoke --live --config configs/provider-smoke.toml --send
uv run tcdata provider smoke --live --config configs/provider-smoke.toml --role mini_verifier --send
```

Bu iki komut yalnız kod içindeki sentetik örneği gönderir. Gerçek kaynak kullanımı için yukarıdaki pilot → selection → explicit live batch sırası ve insan-review kapıları zorunludur.
