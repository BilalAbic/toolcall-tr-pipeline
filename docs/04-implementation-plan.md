# 04 — Uygulama planı ve kabul kapıları

Bu planın Faz 1–4 çekirdeği ve Faz 5'in sözleşme/güvenlik katmanı fixture/property testleriyle uygulanmıştır. DeepSeek V4 çeviri ve OpenAI Responses judge adapterları sabit sentetik smoke girdileriyle ve source-explicit/conflict-free 3 episode / 20 leaf pre-review canary ile canlı endpointte doğrulanmıştır. NVIDIA When2Call'ın test ve train split'leri ile Salesforce xLAM 60k train kaynağı revision-pinned, modelsiz snapshot → ingest → canonical → audit pilotlarından geçti. İnsan adjudication, S400 üyeliği veya release üretilmedi. Faz 6–9 için yetkili insan kararları zorunludur; sistem bunları uydurmaz. Aşamalar atlanmayacak; her fazın kapısı geçmeden sonraki faz üretim verisine açılmayacaktır. Test edilmiş özellikler ile açık maddelerin kanıt tablosu [06 — Uygulama durumu](06-implementation-status.md) belgesindedir.

## Faz 0 — Plan freeze

Teslimatlar:

- karar kaydı,
- canonical conversation sözleşmesi,
- ID/fingerprint sözleşmesi,
- behavior taxonomy,
- prompt/research/eval tasarımı,
- kalite ve release kapıları.

Çıkış ölçütü: kullanıcı onayı ve belgelerin `plan-v1` etiketi.

İlk çalışan sözleşmeler `0.1.0` ile başlar. S30 sonrasındaki geriye uyumlu düzeltmeler `0.2.x`, sonraki pilotlar `0.x` olarak ilerler. `1.0.0`; S400 gold kapısı, migration provası ve bilinen uyumsuzlukların kapanmasından önce verilmez.

## Faz 1 — Temel proje iskeleti

Durum: **uygulandı ve yerel kalite komutlarıyla doğrulanır.** Faz kapısındaki ifadelerden yalnız mevcut kod/test kanıtı bulunanlar tamamlanmış kabul edilir; ayrıntı için durum belgesine bakın.

Kurulacaklar:

```text
pyproject.toml
uv.lock
src/toolcall_tr/
tests/
schemas/
prompts/
configs/
sources/          # gitignore; immutable upstream snapshots
artifacts/        # gitignore; content-addressed derived outputs
docs/
```

Ayrıca ilk günden iki ayrı provenance altyapısı kurulur:

- deterministic content/membership manifest,
- append-only JSONL run/attempt/review olay günlükleri.

Canonical schema'ya ek olarak diagnostic envelope şeması ve sürümlü diagnostic catalog da ilk fazda kurulur. CLI ve testler kararlarını serbest metne değil kararlı diagnostic kodlarına bağlar.

Timestamp ve run ID deterministik manifestin içine konmaz. Artifact önce temp dizinde yazılır, doğrulanır ve atomik olarak yeni içerik-adresli hedefe publish edilir; overwrite yasaktır.

CLI iskeleti:

```text
tcdata source register
tcdata source json-array-to-jsonl
tcdata ingest
tcdata registry build
tcdata canonicalize
tcdata source validate
tcdata source review
tcdata audit duplicates
tcdata select freeze
tcdata translate
tcdata validate
tcdata review
tcdata release build
tcdata inspect
tcdata render
tcdata stats
tcdata diff
tcdata events show
tcdata index rebuild
tcdata memory inspect
tcdata memory promote
tcdata memory conflicts
```

Kapı:

- clean environment `uv sync --frozen`,
- Ruff/Pyright temiz,
- temel CLI smoke,
- schema/model round-trip,
- manifest/event hash-chain ve atomic publish testleri,
- JSONL sharding, streaming scan ve yeniden üretilebilir index testleri,
- yarım/bozuk shard'ın publish edilmemesi ve aynı run'ın atomic-resume testi,
- bilinmeyen diagnostic kodunun fail-closed reddedilmesi ve exit-code sözleşmesi,
- varsayılan akışta hiçbir provider çağrısı yok; canlı smoke/batch yalnız explicit `--live`, non-default config, preflight ve disjoint output ile açılır. İnsan review/selection öncesinde yalnız source-explicit, conflict-free, alias-dışı ve en çok 30 episode'luk pre-review canary egress'i yapılabilir; S400/Gold/release kapalı kalır.

## Faz 2 — Source register ve ID-first ingest

Durum: **uygulandı ve iki gerçek kaynakta kanıtlandı.** When2Call JSONL doğrudan; xLAM 60k ise kaynak JSON dizisi değişmeden immutable JSONL türevine dönüştürülerek işlendi. Her iki kaynak için revision, lisans, file hash ve row count pilot artifactlarına bağlandı.

İki dataset için yalnız kaynak biçimine özel adapter yazılır. Önce snapshot/file manifest, `source_occurrence_id`, snapshot-içi `source_sequence` ve raw hash; strict parse sonrasında varsa native ID üretilir.

Testler:

- byte-offset tabanlı physical occurrence ID,
- snapshot içine satır eklenince eski snapshot kimliklerinin mutation değil yeni snapshot olarak kalması,
- aynı snapshot'ta deterministik sıra ve ID,
- source content mutation yakalama,
- duplicate JSON key, NaN, bozuk Unicode, eksik alan ve büyük kayıt quarantine,
- kaynak dosyaya yazma yapılmadığı kontrolü.

Kapı: kaynak row count + hash doğrulanır; tüm geçerli/hatalı satırlar muhasebeleştirilir.

## Faz 3 — Canonical schema ve tool registry

Durum: **fixture/property testleri ve iki gerçek kaynakta uygulanmıştır.** When2Call tüm erişilebilir splitleri: 27.393 canonical / 559 karantina / dört split birlikte 136 human-review conflict adayı. Training satırları seçilmiş `<TOOLCALL>`, açık clarification ve açık tool-unavailable hedefleriyle dahil edilir; belirsiz metinler ve bozuk tool/argument şemaları karantinada kalır. xLAM 60k: 57.718 canonical / 2.282 karantina / 6 insan-review hard conflict adayı. Karantina kaynak satırını değiştirmez; Gold'a giriş değildir.

Kurulacaklar:

- Pydantic strict canonical modeller,
- JSON Schema 2020-12 artifact,
- RFC 8785 canonicalizer,
- tool semantic normalizer,
- raw/semantic/doc fingerprintleri,
- role/call state machine.

Property-based testler:

- object key sırası hash'i değiştirmez,
- `required`/`enum` sırası semantic hash'i değiştirmez,
- sıra-semantikli listelerin sırası hash'i değiştirir,
- description değişimi structural tool ID'yi değiştirmez,
- default/property/type değişimi tool ID'yi değiştirir,
- call order behavior fingerprint'ini değiştirir.
- yalnız sıra farkı ve topology unknown ise hard conflict yerine order-ambiguity review oluşur.
- desteklenmeyen JSON Schema keyword'ü sessizce düşmez.
- tool-call-only kaynak episode'u `awaiting_tool` olarak geçer; eksik tool result/final otomatik üretilmez.

Kapı: iki datasetin tamamı canonicalize veya gerekçeli quarantine olur; kayıp satır yoktur.

## Faz 4 — Kaynak doğruluğu, duplicate, conflict ve selection

Durum: **altyapı fixture düzeyinde; gerçek exact conflict kanıtı When2Call cross-split (136) ve xLAM 60k (6) üzerinde üretilmiştir.** Explicit Pointer kanıtlı Pass 1, deterministic `source_review`/`source_invalid` routing, exact alias/conflict audit, review-only near-duplicate retrieval, connected-component split guard, append-only human-adjudication logu, karar üretmeyen `review prepare` worklist'i, en çok 30 episode'luk conflict-free pre-review canary ve S400 prefix freeze kod/test yüzeyinde mevcuttur. Pre-review canary, prompt/provider testini insan kabulundan önce yürütür ancak `source_valid`, insan kararı veya S400 freeze üretmez; 2.841 karantina ile 142 conflict için yerel worklist açıktır.

Kurulacaklar:

- source semantic evidence ve `source_valid|source_review|source_invalid` kapısı,
- her expected argument için zorunlu `argument_provenance`,
- yalnız kaynak veya insan adjudication yetkili `acceptable_behaviors`,
- all-record deterministic pass + ranked-frontier blind judge/human pass,
- exact raw/canonical/context/behavior audit,
- cross-source alias,
- behavior conflict quarantine,
- near-duplicate candidate retrieval,
- connected-component split guard,
- katmanlı selection manifesti,
- append-only conflict adjudication eventleri.

Tüm membership, duplicate owner, conflict ve no-repeat indeksleri deterministik JSONL türevleridir. Herhangi bir veritabanı kurulmaz.

Önce küçük etiketli pair setiyle near-duplicate eşikleri ölçülür. Otomatik drop kapalıdır.

Kapı:

- seçilen tüm episode'lar `source_valid`, otomatik source rewrite/relabel sayısı 0,
- `unknown` veya `must_not_infer` argument provenance taşıyan çağrı otomatik valid olamaz,
- judge/model kendi başına yeni bir kabul edilebilir alternatif davranış ekleyemez,
- pahalı source judge'ın yalnız ranked frontier'da tüketilen adaylara çağrıldığı budget testi,
- xLAM argument grounding/tool relevance ve no-tool clarification/unavailable fixture testleri,
- seçki içinde source occurrence/raw/source-episode duplicate yok,
- unresolved conflict yok,
- split leakage yok,
- aynı 400 kayıt yeniden batch diye sunulunca tamamı reddedilir,
- sıralı S400 master manifesti bir kez dondurulur; S30/S100/S250 prefix kümeleridir,
- content/membership manifesti ikinci çalıştırmada byte-identicaldır; run event'i aynı output manifestine referans verir.

## Faz 5 — Prompt ve eval laboratuvarı

Durum: **yerel sözleşme/güvenlik altkümesi ve sınırlı canlı smoke/pre-review canary uygulandı.** Field policy, yalnız leaf segment extraction/host merge, sentinel-bütünlüğü request/response doğrulaması, immutable prompt bundle, kayıtlı-response fake provider, pre-egress scan, exact human-promoted memory, not-sent research policy ve atomic MQM/Wilson/human-only acceptance sözleşmesi fixture testleriyle vardır. DeepSeek V4 çevirici ve OpenAI Responses judge, allow-list'li secret resolver, public HTTPS/no-redirect transport ve preflight arkasında sabit sentetik isteklerle; ayrıca 3 source-explicit/conflict-free When2Call episode'unun 20 leaf'iyle doğrulanmıştır. Automated repair, S400/Gold/release ve insan-review yerine geçecek hiçbir model kararı yoktur.

Üretim API'sinden önce:

- field policy,
- segment extractor/merger,
- sentinel protector,
- translator output schema,
- research policy/schema,
- exact-context accepted-segment JSONL reuse sözleşmesi,
- judge schema/rubrik,
- adversarial eval set,
- fake provider ve recorded-response testleri

kurulur.

Kapı:

- teknik alan invariant testleri %100,
- prompt injection canary geçer,
- boş/truncated/invalid JSON retry testi geçer,
- secret/PII/local-path taraması her egress'ten önce provider çağrısını bloklayabilir,
- research fetch private IP/redirect/MIME/size/time/query bütçelerini fail-closed uygular,
- araştırma tool'u teknik kaydı değiştiremez,
- segment reuse yalnız exact source/context/policy ve insan promotion kararıyla çalışır,
- memory dev/canary/frozen eval koşularında fail-closed kapalıdır,
- production raporunda model-generated ve memory-reused sonuçlar ayrı ölçülür,
- aynı memory key/farklı hedef conflict üretir; fuzzy/partial eşleşme reuse yapmaz,
- unresolved/conflicting research otomatik accepted olamaz,
- judge metrikleri insan gold setinde payda, slice ve güven aralığıyla raporlanır.

Deterministik ürün kapıları:

- structural preservation: %100,
- source/target segment coverage: %100.

İlk 30 için “gözlenen critical/major kabul hatası 0” bir canary sonucudur, gerçek hata oranının istatistiksel kanıtı değildir. Judge otomasyon eşiği daha büyük, önceden dondurulmuş hata-pozitif eval setinde exact-binomial/Wilson güven aralığı ve kategori başına minimum örnekle tanımlanacaktır.

## Faz 6 — 30 kayıt pilotu

Durum: **uygulanmadı; S30 üyeliği veya model attempt'i yoktur.**

Dengeli S400 master manifestinin ilk 30'u kullanılır; pilot sırasında yeni seçim yapılmaz. İki kaynak, tüm action/call-shape sınıfları, farklı tool/domain/uzunluklar ve adversarial örnekler ranking aşamasında dengelenmiştir.

Tüm 30 kayıt şu zincirden geçer:

```text
pre-egress → DeepSeek/reuse → deterministik → {kör GPT mini + kör GPT güçlü} → render/loss-mask → insan
```

Kapı: 30/30 expected episode ID işlenmiş, extra=0, unresolved conflict/research=0, gözlenen critical/major kabul hatası=0. Prompt/config/provider fingerprint değişirse membership korunur fakat attempt kapısı sıfırlanır.

## Faz 7 — 100, 250 ve 400 kapıları

Durum: **uygulanmadı.**

S400 master manifestinden türetilen kümeler strict nested olur:

```text
S30 ⊂ S100 ⊂ S250 ⊂ S400
```

API delta çağrıları yalnız aynı configte henüz accepted olmayan yeni episode ID'ler için yapılır: `+30`, `+70`, `+150`, `+150`. Önceki accepted attempt tekrar çağrılmaz. Her kapı ayrı kalite, token, araştırma, retry, repair ve insan-zaman raporu üretir.

Kapı: tüm 400 kayıt için insan kabulü, trainer chat-template/loss-mask render testi ve release öncesi round-trip doğrulaması. İnsan review; kör değerlendirme, sürümlü rubrik ve calibration/adversarial dilimde çift annotator + adjudication içerir.

Render test matrisi en az şunları kapsar: tool-call-only xLAM, clarification/unavailable text, multi-call, gerçek result+final fixture, JSON-in-JSON escaping, uzun tool şeması, target sınırda max-length, Türkçe I/İ/ı/i ve prior-assistant-turn masking. Target veya gerekli tool schema truncate edilirse silent truncation yoktur.

Her kapı sonrasında hata matrisi JSON olarak üretilir: source, action/call shape, tool ailesi, domain, uzunluk, argument policy, hata kategorisi, research/reuse/repair durumu. S400 membership değiştirilmez; bulgular prompt/policy iyileştirmesine ve 400 sonrası yeni batch sampling'ine yön verir.

## Faz 8 — Ölçekleme

Durum: **uygulanmadı.**

400 sonrasında yeni batch, append-only JSONL olay geçmişindeki source occurrence/raw/source-episode üyelik kimliklerine streaming anti-join edilir. `context` veya `behavior` fingerprint'i tek başına yeni kaydı engellemez; exact birleşimleri duplicate, aynı contextte farklı behavior conflict için kullanılır. Rejected, quarantined ve abandoned occurrence otomatik “yeni” sayılmaz; açık supersede kararı gerekir.

Ölçekleme öncesi kararlar metrikle verilir:

- GPT güçlü judge kapsam oranı,
- insan örnekleme oranı,
- research trigger eşiği,
- near-duplicate retrieval teknolojisi,
- shard/row-group boyutu.

İnsan örneklemeye düşülürse yalnız tek tek insan kabulü alan satırlar gold olabilir; diğer başarılı satırlar silver release'e gider. Otomatik gold kabulü v1'de yoktur.

## Faz 9 — JSONL release ve isteğe bağlı Parquet/HF doğrulama

Durum: **uygulanmadı; release veya Hugging Face/Parquet çıktısı yoktur.**

Release çıktıları:

```text
canonical/*.jsonl.zst
data/*.parquet       # yalnız HF export açılırsa
MANIFEST.json
README.md / Dataset Card
schema/*.json
reports/quality.json
reports/provenance.json
```

Kapı:

- canonical JSONL schema ve full-row JCS hash doğrulaması,
- Parquet export açılırsa explicit Arrow schema ve JSONL↔Parquet full-row deep equality,
- stable ordered logical dataset hash,
- shard/file hash ve row count,
- `load_dataset(..., streaming=True)` smoke,
- Hugging Face Viewer validation,
- hedef model ailelerinde chat-template render ve loss-mask doğrulaması,
- model-family training view manifesti ve `render_config_id` pinlemesi,
- release secret/PII/local path scanına ek olarak önceki pre-egress tarama kanıtı,
- lisans/provenance doğrulaması.

## İlk uygulama sprinti

Tamamlanan temel Faz 1–5 altyapısı ve sınırlı canlı operasyon yüzeyidir:

1. bağımsız proje iskeleti ve lockfile,
2. canonical Pydantic/JSON Schema,
3. deterministic JSON manifest + append-only JSONL event altyapısı,
4. source manifest ve occurrence-ID-first ingest,
5. fixture, When2Call ve xLAM 60k source adapterları ile JSON-dizi → immutable JSONL dönüşümü,
6. tool registry/fingerprint,
7. property-based testler,
8. `inspect` ve `stats` CLI'sı.

Gerçek çeviri/judge API kanıtı sentetik smoke ile source-explicit/conflict-free 3 episode / 20 leaf pre-review canary sınırındadır. Kimlik, veri formatı, immutable publish ve temel tekrar-almama garantileri fixture/property testleriyle doğrulanır; iki gerçek kaynak için tam canonical/karantina muhasebesi modelsiz pilotta tamamlanmıştır.

## Plan değişiklik kuralı

- Her schema/prompt/normalizer sürümü immutabledır.
- Diagnostic kodunun anlamı aynı catalog sürümünde değiştirilemez; yeni anlam yeni kod veya yeni catalog sürümü gerektirir.
- Bir karar değişirse mevcut artifact mutate edilmez; yeni sürüm ve migration notu oluşturulur.
- Prompt değişiklikleri tek tek yapılır; günlük karşılaştırma dev+canary üzerinde, frozen gold yalnız promotion kararında yürütülür.
- Yeni teknoloji yalnız ölçülmüş bir darboğazı çözüyor veya güvenceyi artırıyorsa eklenir.
