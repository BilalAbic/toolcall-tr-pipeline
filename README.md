# Toolcall TR Pipeline

Bu proje, İngilizce tool-calling ve no-tool konuşma verilerini yüksek kaliteli Türkçe eğitim verisine dönüştürmek için tasarlanmış bağımsız bir veri hattıdır.

Kod deposu: [BilalAbic/toolcall-tr-pipeline](https://github.com/BilalAbic/toolcall-tr-pipeline).

## Özet

Faz 1–5 veri hattı, fail-closed canlı çeviri/judge yüzeyleri ve bounded otomasyon katmanı uygulanmıştır. `pilot run`, kaynağı değiştirmeden snapshot → ingest → canonical → audit zincirini yürütür; `automation run` conflict-free adayları seçer, çevirir, önce mini judge ile değerlendirir ve yalnız non-pass ile deterministik denetim örneklemini güçlü judge’a yükseltir. İnsan onayı bekleyen Hugging Face JSONL paketi üretir; her Gold/release kaydı açık insan kabuluna bağlıdır.

| Alan | Mevcut durum |
|---|---|
| Kaynak pilotları | When2Call ve xLAM 60k revision-pinned, modelsiz işlendi. |
| Policy coverage | 85.111 canonical episode, 849.064 çeviri segmenti, 0 unresolved policy error. |
| İnsan review kuyruğu | 2.983 açık görev yayımlandı: 2.841 canonical karantina ve 142 unresolved conflict. |
| Pre-review canlı canary | When2Call'dan 3 conflict-free episode / 20 leaf çevrildi; mini judge 20/20 `pass`, 0 finding. |
| Otomatik canlı 50’lik koşu | Doğrulanmış `prompt-0.3.0` + Flash 6 worker regresyonunda 50 adaydan 46’sı çevrildi; 547 leaf’in 541’i kabul edildi ve 40 satırlık HF review paketi oluştu. Tahmini maliyet $0.556529. v0.4 deneyi bunu geçemediği için üretimde kapalıdır. |
| İnsan review / Gold | Model ve deterministik kontrollerden sonra son kabul kapısı; pre-review canary Gold/S400/release üretmez. |

## Tasarım referansları

Bu bağımsız pipeline, aşağıdaki kardeş depolardan tasarım desteği alır. Kaynak
snapshotları veya onların eğitim verileri bu depoya kopyalanmaz; burada yalnız
uyarlanmış ve yeniden test edilmiş sözleşmeler uygulanır.

- [turkish-tool-calling-dataset](https://github.com/BilalAbic/turkish-tool-calling-dataset): aşamalı `S30 → S100 → S250 → S400` seçim yaklaşımı, tekrar-önleme/append-only ledger fikri, teknik alanların deterministik korunması ve çok kapılı kalite akışı için referans oldu.
- [magibu-toolcall](https://github.com/BilalAbic/magibu-toolcall): schema-first registry ve canonical audit yaklaşımı, provenance odaklı kayıtlar, teknik alanların modelden bağımsız doğrulanması ve insan kabulunun model kararından ayrı tutulması için referans oldu.

Bu projede bu ilkeler; içerik-adresli artifactlar, hash zincirli event logu, strict JSON Schema, host-side merge, insan-kapılı selection/release ve kapsamlı testlerle uygulanmıştır.

İç mekanizmaların **W3C PROV-DM**, **C2PA**, **MLCommons Croissant**,
**Datasheets for Datasets / Model Cards**, **DVC** ve **RO-Crate** gibi
tanınmış veri-provenance standartlarına haritası ile tam geçiş yol haritası
için [09 — Mimari ve kaynaklar](docs/09-architecture-and-references.md)
belgesine bakın.

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
- canlı preflight: payloaddaki secret, PII ve local-path bulgularında transport öncesi ret; sabit synthetic smoke ve küçük, conflict-free pre-review canary için explicit `--live` yürütmesi,
- explicit `pilot run`: kaynakla kesişmeyen output altında snapshot/ingest/canonical/audit; quarantine veya çözülmemiş conflict olduğunda fail-closed,
- canonical karantina ve exact-conflict kanıtını sıralı, karar üretmeyen insan review görevlerine dönüştüren `review prepare`; reviewer-authored tek-kayıt JSONL kararını doğrulayan append-only review komutları; Gold release için model verdict + insan kabul bağlantısı zorunlu,
- hash-only provider attempt kayıtları: ham request/response/credential tutulmaz; temel adapter çağrıları tek-attempt’tır. Otomasyon yalnız doğrulanmış response/HTTP transient sınıflarında bir kez Flash → Pro fallback uygular; teslimatı belirsiz ağ hatası yeniden gönderilmez,
- explicit pre-review canary: source-explicit, policy-covered, conflict-free ve exact-alias dışı en çok 30 episode seçimi; immutable çeviri/judge inputları, otomatik retry, Gold/S400/release veya insan kararı yok,
- explicit canlı çeviri: policy-izinli leaf başına immutable claim/attempt/checkpoint, kaynak rehash ve host-only merge; otomatik retry, Gold ve release yok,
- explicit canlı judge: tam-leaf content hash'iyle bağlı strict source/target girdileri, immutable result/attempt manifestleri ve `gold_release_allowed=false`,
- yalnız enjekte edilmiş renderer ve tokenizer ile çalışan render/loss-mask sözleşmesi: pinli render config, tek final-assistant payload aralığı, teknik yapı hash'i ve truncation/uyuşmazlıkta fail-closed; chat-template veya tokenizer indirme yok,
- insan onayı bekleyen Hugging Face review paketi: doğrudan yüklenebilir `data/train.jsonl` (`messages` + `tools`), `dataset_info.json`, Dataset Card ve hashli `manifest.json`; Gold/HF publish yetkisi üretmez,
- `source register`, `source validate`, `source json-array-to-jsonl`, `source evidence`, `ingest`, `registry build`, `canonicalize`, `audit duplicates`, `audit near-duplicates`, `select freeze`, `pilot run`, `canary prepare`, `canary evaluation-inputs`, `translate`, `evaluation run`, `automation run`, `review prepare`, `review submit-evaluation`, `review submit-conflict`, `release build`, `release validate`, `inspect`, `stats`, `events show` ve `diagnostics` CLI komutları.

Gerçek eval UI, render ve HF publish komutları hâlâ kapalıdır. Review/release komutları yalnız dışarıdan sağlanan insan kararlarını doğrular; karar üretmez. Canlı çeviri/judge çağrıları yalnız explicit `--live`, non-default config, ayrı output kökü ve preflight ile gerçekleşir; kaynak dataset asla varsayılan olarak gönderilmez.

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

Pilot kanıtı olmadan çeviri veya Gold release açılmaz. `automation run`, pilot canonical artifact’larından yalnız source-explicit, policy-covered, conflict-free/alias-dışı cohort seçer; karantina ve conflict kayıtlarını değiştirmez. Human-review ve release komutları, gerçek model verdict JSONL ve reviewer tarafından yazılmış karar JSONL'si gerektirir; bunlar olmadan Gold/publish fail-closed kalır.

2026-08-13'te [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call) revision `0582f7749df63a96fdc3070932e83e72396ace53` altındaki tüm erişilebilir veri split'leri salt-okunur pilotta işlendi. `train/sft` ve `train/pref` içindeki seçilmiş assistant hedefleri artık dahil edilir: açık `<TOOLCALL>` işaretleri, açık ek-bilgi istemleri ve açık tool-unavailable metinleri canonical'a alınır; belirsiz veya schema/argument açısından geçersiz kayıtlar karantinada kalır. 27.952 satırdan 27.393 canonical survivor üretildi; 559 gerekçeli karantina var. Dört split birlikte 2.917 exact duplicate group ve 136 human-review conflict adayı üretti. Bu modelsiz bir işlemdir; hiçbir gerçek kaynak satırı provider'a gönderilmedi, Gold/release veya insan kararı üretilmedi.

Kullanıcının Hugging Face gated erişimiyle [Salesforce xLAM 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) train kaynağı revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866` ile indirildi. 96.1 MB JSON dizi, tam dosya hash'i bağlı immutable JSONL'ye dönüştürüldü ve tamamı modelsiz pilotta işlendi: 60.000 source-valid satırdan 57.718 canonical kayıt, 2.282 gerekçeli karantina ve insan review bekleyen 6 hard-conflict adayı oluştu. Pilot kaynak veya karantina kayıtlarını değiştirmedi ve provider egress'i yapmadı; daha sonraki bounded automation koşusuna yalnız conflict-free canonical candidate leaf'leri explicit olarak girdi.

Gözden geçirilmiş field policy, When2Call+xLAM toplam 85.111 canonical episode üzerinde ağsız doğrulandı: 849.064 çevrilebilir segment ve 0 çözülmemiş policy hatası. Tool/parameter açıklamaları çevrilebilir; tüm tanımsız argument path'leri `copy_exact` kalır ve modele gönderilmez. Bu kapsam, yalnız source-explicit ve conflict-free pre-review canary için küçük canlı testlere izin verir; S400, Gold ve release insan kabulundan önce kapalıdır.

`translation-prompt-0.3.0`, modelin kaynak leaf içindeki talimatları veri olarak ele almasını, teknik alanların host-owned kalmasını, sentinel’ların byte-for-byte korunmasını, URL/domain gibi gömülü teknik span’ları korurken çevresindeki doğal dili çevirmesini, belirsiz terimde `research_needed` dönmesini ve JSON-only yanıtını açıkça güçlendirir. v0.4 aday prompt’u, v0.3 regression’ındaki altı strong-judge bulgusundan dar karşıt örneklerle üretildi fakat aynı 50-episode cohort’ta 535 accepted leaf ve bir strong `provider_response_invalid` sonucu verdi; v0.3’ün 541 accepted leaf / 0 unavailable sonucunu geçemediği için aktif değildir. Prompt bundle artık `candidate`, `validated` veya `retired` promotion durumu taşır; canlı çeviri yalnız `validated` prompt’u kabul eder. Canlı örnekteki `reverse_input:/input_value` ve `qrcode:/data` doğal dil görünse de aynı yollar başka kayıtlarda çok-türlü değer veya identifier taşıdığından yeni argument çeviri istisnası açılmamıştır; tüm argument path’leri `copy_exact` kalır. Önceki `0.2.0` canary’si When2Call’da 3 episode / 20 leaf’i DeepSeek V4 Flash ile çevirdi; GPT-5.4-mini 20/20 pair için `pass`, 0 finding verdi. Bu model triage kanıtıdır; hiçbir satır `gold_eligible` değildir ve insan incelemesi son kabul kapısıdır.

Bu iki pilotun karar gerektiren kanıtları, hiçbir kaynak satırı veya karar içermeyen tek bir yerel review kuyruğunda yayımlandı: `manifest_a52302ac6fac9d362db23109b2ee20bd762897ad653cf143d2c7b81ba9cc0c8d`. Kuyruk 2.841 canonical-karantina ve 142 conflict-adjudication görevi taşır. Her conflict satırı mevcut `review submit-conflict` akışına, her karantina satırı ise yalnız insan onaylı adapter/policy düzeltmesiyle yeniden pilotlama gerektiren remediation işine bağlanır; otomatik drop veya kabul yoktur.

Pilot ve selection kapıları geçtikten sonra sınırlı bir batch için canlı adımlar şunlardır:

```powershell
uv run tcdata translate <canonical.jsonl> --output <disjoint-translation-output> --config configs/provider-smoke.toml --field-policy configs/field_policy.toml --prompt configs/prompt_bundle.toml --live
uv run tcdata evaluation run <live-evaluation-input.jsonl> --output <disjoint-evaluation-output> --config configs/provider-smoke.toml --role mini_verifier --run-id <run-id> --live
```

İki komut da otomatik Gold/release yapmaz ve ham içerik ya da anahtar tutmayan attempt manifestleri üretir.

## Otomatik aday → HF review paketi

`automation run`, insan incelemesini kayıt-bazlı erken kapı olmaktan çıkarıp yayın öncesindeki son kapıya taşır. Aynı route tekrar çalıştırılırsa immutable checkpoint kullanılır; response-contract/HTTP transient veya açık politika/research ihtiyacında DeepSeek Flash’tan Pro’ya bir kez geçilir. Mini judge her leaf’i değerlendirir; mini non-pass kayıtlar ile `--strong-pass-sample-percent` tarafından seçilen deterministik mini-pass örneklemi güçlü judge’a gider. Örneklem dışındaki mini `pass` yeterlidir; escalation’daki güçlü judge karar verir. Ağ teslimatı belirsizse veya güçlü judge `needs_human_review`/ulaşılamaz sonucu verirse yeniden gönderme yapılmaz; kayıt HF paketine alınmaz, ama batch devam eder.

```powershell
uv run tcdata automation run <canonical-1.jsonl> <canonical-2.jsonl> `
  --conflict-audit <when2call-audit.json> --conflict-audit <xlam-audit.json> `
  --output <disjoint-output> --config configs/provider-smoke.toml `
  --episodes 1000 --max-segments 14000 `
  --strong-pass-sample-percent 2 --live
```

İsteğe bağlı `--source-row-cap` yalnız düşük maliyetli canary için her inputta deterministik pencere açar; 1.000’lik üretim için verilmez. Çıktıdaki `hf-review-package/<package-id>/data/train.jsonl` doğrudan HF JSON loader’a uygundur, ancak manifest `pending_human_approval` ve `publish_allowed=false` olarak kalır. Akışın ayrıntılı failover ve karar matrisi [08 — Otomatik review pipeline](docs/08-autonomous-review-pipeline.md) belgesindedir.

Uzun koşu ayrı bir PowerShell penceresinde ya da `Start-Process` ile arka planda başlatılabilir. Aynı `--output` köküyle komutu tekrar çalıştırmak, mevcut translation route receipt'lerini ve judge checkpoint'lerini kullanır; tamamlanmış satırlar yeniden provider'a gönderilmez. Durumu ve run içi token/maliyet toplamını ağ/API erişimi olmadan izlemek için:

```powershell
uv run tcdata automation status <output-root> --config configs/provider-smoke.toml
```

Eski immutable artifact'ler silinmez. Yeni bir ana üretim koşusu için yeni, boş ve tarih/run-id içeren bir output kökü seçilir; yalnız yarıda kalan **aynı** koşu aynı kökle devam ettirilir. Bu ayrım, tekrar egress'i ve kanıt karışmasını engeller.

İkinci ve sonraki üretim batch'leri aynı adayları yeniden seçmemelidir. Yeni output köküyle beraber `--candidate-offset` değerini önceki aday sayısı kadar artırın; örneğin ilk 1000 aday için `0`, sonraki 1000 aday için `1000`. Offset immutable candidate manifestine yazılır ve kaynaklar değiştirilmez.

Ana koşudan önce aynı argümanlarla `tcdata automation plan` çağrısı yapılabilir. Bu yalnız immutable candidate manifestini yazar, gerçek episode/leaf sayısını gösterir ve hiçbir provider/API çağrısı yapmaz; planın output kökü daha sonra `automation run --live` ile aynen kullanılır.

Canlı CLI her tamamlanan aşamada provider/model bazında gerçek sağlayıcı yanıtından gelen input/cache/output tokenlarını, istek sayısını ve USD fiyat kartına göre tahmini kümülatif maliyeti gösterir. Bu kayıtlar `provider-usage/` altında yalnız sayaç ve hash kimliğiyle tutulur; ham istek, yanıt veya anahtar içermez. `configs/provider-smoke.toml` Flash için 6, mini için 4, güçlü judge için 2 worker; mini için 1.8M ve güçlü judge için 225k ihtiyatlı günlük token uyarı eşiği tanımlar. Bu eşikler tek koşunun tüketimini görünür kılar; sağlayıcının hesap-geneli günlük kotasının yerine geçmez. `--workers` verilirse tüm role limitlerini açıkça override eder. `--strong-pass-sample-percent` (varsayılan `2`) güçlü judge maliyetini görünür, deterministik ve ayarlanabilir tutar; mini non-pass kayıtlar her zaman escalation’dadır.

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
