# 03 — Prompt, araştırma ve kalite tasarımı

Durum: **Faz 5 sözleşme/güvenlik tabanı ve sınırlı canlı yürütme yüzeyi uygulanmıştır.** `field_policy.py` explicit leaf extraction ve host-side merge'i; `translation_contract.py` segment-only request/response, sentinel, NFC ve coverage kontrolünü; `egress_guard.py` secret/PII/local path/private endpoint taramasını; `eval_contract.py` atomic MQM bulguları, Wilson %95 coverage raporu ve human-only gold kabulünü; `prompt_contract.py` immutable katmanlı prompt bundle'ı uygular. `secure_transport.py`, allow-list'li `.env` resolver ve DeepSeek/OpenAI adapterları yalnız explicit `--live`, non-default config, preflight ve disjoint output ile çalışır; sentetik çeviri/judge smoke'ları canlı endpointlerde doğrulanmıştır. `translation_memory.py` yalnız human-promoted full segmentleri tüm exact context/policy anahtarıyla bulur ve hedef conflictinde fail-closed kalır. `research_policy.py` deterministik risk routing ile public-HTTPS/bütçeli not-sent request/evidence metadata sağlar, fetch içermez. Gerçek kaynak model egress'i, automated repair, insan review ve Gold release hâlâ kapalıdır; eval verdict'i yalnız triage'dır.

## 1. Ana ilke

Prompt teknik yapıyı korumayı söyler; pipeline ise teknik yapıyı modele hiç ürettirmeyerek gerçekten korur.

```text
canonical episode
  → pre-egress güvenlik
  → çevrilebilir leaf extraction
  → korumalı span maskeleme
  → bağımsız terminoloji risk router
  → gerekirse sınırlı araştırma
  → DeepSeek çeviri taslağı
  → host-side merge
  → deterministik validator
  → {kör GPT mini adversarial verifier + kör GPT güçlü semantik judge}
  → gerekirse tek kontrollü repair
  → training-template render ve loss mask
  → insan kabulü
```

## 2. Field policy

Her JSON Pointer yolu aşağıdaki sınıflardan birine girer:

- `copy_exact`
- `translate`
- `translate_if_natural_language`
- `omit_from_model_input`
- `manual_policy_required`

Örnek politika:

| Alan | Politika |
|---|---|
| user/assistant doğal dil `content` | `translate` |
| tool/function `description` | `translate` |
| parameter `description` | `translate` |
| function/parameter adı, role, ID | `copy_exact` |
| enum/default/type/required | `copy_exact` |
| call/message sırası | `omit_from_model_input` + host merge |
| argument ID/URL/path/enum/number/boolean | `copy_exact` |
| açıkça serbest metin olarak onaylanmış argument path | `translate` |
| policy'de adı geçmeyen herhangi bir argument path | gözden geçirilmiş global fallback: `copy_exact` |
| reasoning/thinking | kaynakta yoksa `null`, üretilmez |

Argument değerleri varsayılan olarak modele gönderilmez: `configs/field_policy.toml` içindeki tek izinli global fallback `*` / `/*` yalnız `copy_exact` olabilir. Bir argument ancak isimli tool + exact JSON Pointer ile ayrıca gözden geçirilip `translate` yapılabilir; schema formatı, enum üyeliği ve path güvenlik denetimi yine uygulanır. Böylece gerçek tool kayıtları eksik policy nedeniyle durmaz, fakat teknik veya kullanıcı girdisi niteliğindeki argumentlar yanlışlıkla çevrilmez. Bu yapı bir sözcük sözlüğü değil, alanın veri tipini ve işlevini tanımlayan sürümlü bir sözleşmedir.

## 3. Çevirici system prompt katmanları

Tek büyük ve tekrarlı system prompt yerine derlenen kısa katmanlar kullanılır:

1. `core_contract`: görev, başarı ölçütü, kaynak metnin talimat değil veri olduğu kuralı.
2. `field_policy`: hangi segmentlerin çevrileceği.
3. `fidelity_contract`: anlam, niyet, olumsuzluk, modalite, sayı, birim, tarih, özel ad, register.
4. `protected_span_contract`: segment/occurrence sentinellerini aynı konum ilişkisiyle döndürme.
5. `terminology_protocol`: emin değilse tahmin etmeme ve araştırma isteği.
6. `output_contract`: yalnız beklenen JSON segmentleri.
7. `contrastive_examples`: yalnız ölçülmüş hatalara yönelik 2–4 kısa karşıt örnek.

Sabit prefix prompt caching için değişmeden kalır. Record'a özgü segmentler en sona eklenir. Yalnız gerçek invariants `MUST/NEVER` kullanır; tekrar eden ve sonucu değiştirmeyen talimatlar prompttan çıkarılır.

## 4. Çevirici giriş/çıkış sözleşmesi

Giriş:

```json
{
  "episode_id": "...",
  "input_variant_id": "...",
  "source_language": "en",
  "target_language": "tr",
  "segments": [
    {
      "segment_id": "seg_001",
      "path": "/conversation/0/content",
      "source_text": "...",
      "protected_tokens": [
        {"token": "⟪S001_P001⟫", "occurrence": 1}
      ]
    }
  ],
  "terminology_evidence": []
}
```

Çıkış:

```json
{
  "status": "translated",
  "segments": [
    {
      "segment_id": "seg_001",
      "target_text": "...",
      "research_needed": false,
      "uncertainty_tags": []
    }
  ],
  "term_queries": []
}
```

Model hash hesaplamaz, kaynak JSON ağacını yeniden kurmaz ve yeni segment ekleyemez. DeepSeek JSON Output geçerli JSON üretmeye yardımcı olur fakat yerel JSON Schema garantisinin yerine geçmez ve boş içerik ihtimali vardır. Parse/schema/coverage kontrolü ve sınırlı retry zorunludur. [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)

## 5. Korumalı spanlar

Kod, URL, placeholder, CLI flag, JSON pointer ve identifier'lar segment ve occurrence'a bağlı sentinel ile maskelenir:

```text
https://example.com → ⟪S001_P001⟫
--dry-run           → ⟪S001_P002⟫
${user_id}          → ⟪S002_P001⟫
```

Validator yalnız toplam sentinel kümesini değil, hangi segmentte kaçıncı occurrence'ın bulunduğunu ve çevreleyen noktalama/ek bağlanma politikasını da doğrular.

`copy_exact` alanında byte equality aranır. Doğal dil segmentindeki tarih/sayı/birim için biçim değil, parse edilmiş semantik değer eşitliği aranır; Türkçe locale biçimi ayrı policy ile izinli olabilir.

## 6. Pre-egress güvenlik

DeepSeek, OpenAI veya search çağrısından **önce**:

- secret/token/credential,
- PII,
- local/UNC path,
- private endpoint/host,
- lisans ve provider-egress politikası

taraması yapılır. İhlal, redact edilemiyorsa provider çağrısını bloklar. Araştırmaya tüm kayıt değil yalnız terim ve anlam ayrımı için gerekli en küçük, redacted bağlam gönderilir. Ham kayıt veya kullanıcı tarafından sağlanan URL otomatik olarak açılmaz.

### Bağlamlı tam-segment translation memory

Sözlük veya veritabanı kullanılmaz. Memory yalnız insan tarafından kabul edilip ayrıca reuse için promote edilmiş tam segmentlerden JSONL olarak üretilir:

```json
{
  "schema_version": "segment-memory-0.1.0",
  "memory_entry_id": "mem_...",
  "segment_source_sha256": "...",
  "source_text": "...",
  "target_text_sha256": "...",
  "field_policy_version": "...",
  "argument_policy_version": "...",
  "locale_policy_version": "...",
  "presented_context_fingerprint": "...",
  "tool_scope": ["tool_..."],
  "decision_action": "tool_call",
  "target_text": "...",
  "origin_translation_config_id": "...",
  "human_review_event_id": "...",
  "state": "active"
}
```

Lookup şu alanlarda exact eşleşir: source UTF-8 bytes hash, field/path class, field ve argument policy, locale policy, presented context, tool scope ve decision action. Prompt hash lookup anahtarı değildir; çünkü memory model çıktısı değil insan-promoted curated artifact'tır. Bununla birlikte origin prompt/config provenance olarak tutulur.

Kurallar:

- Kelime, alt-parça, normalize edilmiş veya fuzzy eşleşme reuse yapmaz.
- PII/secret içeren segment memory'ye promote edilmez.
- Aynı lookup key için birden fazla farklı aktif Türkçe hedef varsa `memory_conflict`; otomatik seçim yoktur.
- Düzeltme silme/overwrite değildir; `promoted`, `superseded`, `revoked`, `conflict_resolved` JSONL event'i yazılır.
- Dev, canary, frozen_gold ve model/prompt ablation koşularında memory kapalıdır; eval sızıntısı önlenir.
- Production reuse yine segment/invariant/judge ve geçerli release insan kapılarından geçer; gold etiketi kazandırmaz.
- Yeni episode provenance'ına memory entry ve reuse event ID eklenir.

Bu JSONL görünümü silinirse insan review ve memory event'lerinden yeniden üretilebilir.

Her yeni translation config için iki hat ayrıdır:

- `evaluation_lane`: memory zorunlu kapalı; model/promptun gerçek performansı ölçülür.
- `production_lane`: config promotion sonrasında curated memory açık olabilir; generated/reused kalite, maliyet ve hata oranları ayrı raporlanır.

## 7. Terminoloji risk router ve durum makinesi

Araştırma yalnız çeviricinin özbildirimine bağlı değildir. Aşağıdakilerden biri tetikleyebilir:

- kısaltma/ürün/standart adını bulan deterministik risk kuralları,
- alan-bağımlı çok anlamlılık için küçük classifier,
- argument/field policy belirsizliği,
- çeviricinin `research_needed` çıktısı,
- judge'ın terminology/research bulgusu,
- accepted evidence memory ile çelişki.

Bu, statik karşılık sözlüğü değildir; yalnız “araştırma gerekiyor mu?” yönlendirmesidir.

Durumlar:

```text
not_needed
  veya
needs_research → researched → retranslate
                        ↘ conflicting → human_review
                        ↘ unresolved  → human_review/quarantine
```

`conflicting` veya `unresolved` otomatik accepted olamaz.

## 8. Araştırma ajanı ve fetch sınırları

Araştırmacı çeviriyi düzenlemez; kanıt sidecar'ı döndürür. Kaynak önceliği: dataset/tool bağlamı → resmî vendor dokümanı → standart kuruluşu → resmî kurum.

```json
{
  "term": "...",
  "resolution": "resolved",
  "candidates": [
    {
      "target": "...",
      "source_url": "https://...",
      "source_type": "vendor_docs",
      "evidence_span": "...",
      "confidence": "high"
    }
  ]
}
```

V1 fetch politikası:

- yalnız HTTPS ve genel internet hostları,
- private/link-local/loopback IP ve non-HTTP protokoller bloklu,
- her redirect sonrası host/IP yeniden doğrulanır,
- HTML/plain-text/PDF allowlist; executable ve beklenmeyen MIME bloklu,
- sayfa boyutu, timeout, redirect, sorgu ve kaynak sayısı bütçeli,
- sayfa talimatları güvenilmeyen veri; yürütülmez,
- forum/blog/SEO içeriği karar kanıtı olamaz,
- host final URL, retrieval zamanı, response/content hash ve kısa gerçek kanıt span'ını kendisi kaydeder,
- modelin verdiği URL veya evidence summary tek başına kanıt sayılmaz,
- araştırma çıktısı teknik alanı, sayıyı, decision veya tool davranışını değiştiremez.

Başlangıç bütçesi config'te açıkça pinlenir: episode başına en fazla 3 sorgu, 5 kaynak, 60 saniye ve belirlenen token limiti. Limit aşımı insan kuyruğudur; gizli ek arama yapılmaz.

## 9. Deterministik validator

LLM judge'dan önce:

- çıktı schema uyumu,
- birebir segment ID kümesi,
- segment/occurrence sentinel eşleşmesi,
- function/parameter/enum exact equality,
- rol, mesaj ve tool-call sırası,
- call ID ve tool result bağlantısı,
- arguments JSON ve parameter schema uyumu,
- boolean/null/number tipleri,
- sayı/tarih/birim/URL/placeholder policy uyumu,
- yasak yeni alan veya açıklama,
- Türkçe doğal dil alanlarında NFC

kontrol edilir. Deterministik fail judge'a gönderilmez.

## 10. Model kalite kapısı

### Mini adversarial verifier

Deterministik validator'ı tekrar etmez. Kör biçimde özellikle olumsuzluk, kiplik, eksiltme, akıcı ama yanlış çeviri, clarification/tool-call sınırı ve kaynakta olmayan yardımcı açıklama ekleme gibi ucuz semantik canary'leri inceler.

### Güçlü semantik judge

Kaynak–hedef eşdeğerliği, niyet, terminology, ekleme/eksiltme, clarification ve tool davranışını tam rubrikle inceler.

İki judge birbirinin kararını görmez. İlk 400'de deterministik kontrolden sonra ikisi de kör ve birbirinden bağımsız girdilerle çalışır; gecikmeyi azaltmak için paralel çağrılabilirler. Aynı sağlayıcı ailesi olduklarından bunlar “bağımsız iki yargıç” kabul edilmez. Anlaşmazlık insana gider. Mini'nin güçlü judge'ı atlatan maliyet filtresine dönüşmesi ancak 400 sonrası kalibrasyon kararıdır.

OpenAI çıktıları strict Structured Outputs ile alınır; biçim garantisi semantik doğruluğun yerine geçmez. [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## 11. MQM tabanlı atomik rubrik

Kategoriler:

- `accuracy.mistranslation`, `accuracy.omission`, `accuracy.addition`, `accuracy.untranslated`
- `terminology`
- `fluency.grammar`, `fluency.spelling`, `fluency.punctuation`
- `locale_convention`, `style.register`
- `tool_semantics`, `protected_content`, `research_provenance`

Şiddet:

- `critical`: tool/çağrı/güvenlik/karar davranışı değişmiş.
- `major`: niyet veya önemli anlam yanlış; ciddi ekleme/eksiltme.
- `minor`: anlamı değiştirmeyen doğallık, imla veya register sorunu.

Tek 1–10 puanı yoktur. Judge, enum hata kodu + kategori + şiddet + segment/span + kısa kanıt üretir. Model grader insan etiketleriyle kalibre edilmeden otomatik kabul yetkisi alamaz. [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

## 12. Repair protokolü

- Otomatik retry bütçesi `0`dır; parse/schema/transient provider hatası yalnız append-only kayıtta `manual-retry-candidate` olarak sınıflanır ve insan/operatör onayı olmadan tekrar gönderilmez.
- Semantic fail aynı promptla kör retry almaz.
- Judge serbest talimat yerine enum hata kodu, segment ID ve span verir.
- Repair modeline yalnız hatalı segment, kaynak segment, doğrulanmış evidence ve izinli hata kodu gönderilir.
- Diğer segmentler immutable kalır; host değişmediklerini hash ile doğrular.
- En fazla bir semantic repair attempt'i; sonra tüm kapılar baştan.
- İkinci semantic fail insan/reject kuyruğuna gider.
- Tüm denemeler append-only saklanır.

## 13. İnsan gold protokolü

- Gold'a girecek her satır insan tarafından tek tek kabul edilir.
- Reviewer provider/model adını ve judge verdict'ini görmeden kaynak–hedefi değerlendirir.
- Calibration ve adversarial dilim çift kör iki annotator tarafından etiketlenir; anlaşmazlık üçüncü adjudicator ile çözülür.
- Reviewer guideline/rubric sürümü ve adjudication event'i kaydedilir.
- Inter-annotator agreement kategori ve şiddet bazında raporlanır.
- Örneklemeli insan kontrolü kullanılan otomatik kayıtlar yalnız silver olabilir.

İnsan review sırasında reviewer bir segmenti kabul edebilir fakat memory reuse'a promote etmeyebilir. Promote kararı ayrıca PII, context-specific ifade, geçici ürün adı ve birden fazla geçerli Türkçe karşılık risklerini kontrol eder.

## 13.1 Training-template render ve loss mask kapısı

Canonical kayıt modelden bağımsızdır. Trainer adapter her hedef model ailesi için ayrı `render_config_id` ile training view JSONL üretir. Config; tokenizer/revision, chat-template exact hash, tool serialization, BOS/EOS, generation prompt, max length, truncation ve label policy'yi kapsar.

V1 supervision politikası:

- Yalnız son hedef assistant mesajı label alır.
- Önceki system/developer/user/tool ve assistant mesajları `-100`/masked olur.
- xLAM episode'unda hedef tool-call serialization'ının function name ve arguments kısmı label kapsamındadır.
- Final-answer episode'unda yalnız hedef assistant final cevabı label kapsamındadır.
- Model-family control ve EOS tokenlarının supervise edilmesi adapter policy'sinde açıkça sürümlenir.

Fail durumları:

- Hedef token aralığı boş veya birden fazlaysa,
- target assistant ya da gerekli tool tanımı truncate olmuşsa,
- arguments parse→render→parse eşit değilse veya iki kere JSON-string olmuşsa,
- tool-call/result ID bağlantısı kaybolmuşsa,
- conversation prefix farklı içerikle yeniden yazılmışsa,
- Unicode, role/control token veya label-mask golden testleri bozulmuşsa.

`tcdata render --id ... --adapter ... --show-mask` rendered metni, token indekslerini, yarı-açık supervised ranges ve truncation raporunu gösterir. Training view canonical satırı değiştirmez; adapter/config değişince yeniden üretilir. Bu kapı insan incelemesinden önce çalışır.

## 14. Eval seti ve istatistik

Setler:

- `dev`: günlük prompt geliştirme.
- `adversarial_canary`: her değişiklikte kritik regresyon.
- `frozen_gold`: yalnız promotion kararı; sonuçları geliştirmeyi etkiledikten sonra yeni holdout rotasyonu gerekir.
- `shadow`: gerçek kullanımda yeni bulunan hatalar; sonraki dev/canary sürümüne kontrollü eklenir.

Zorunlu dilimler: Türkçe I/İ/ı/i, key/enum, iç içe JSON/escape, placeholder, olumsuzluk ve `must/should/may`, tarih/ondalık, alan-bağımlı terimler, prompt injection, akıcı ama eksik çeviri, uydurma açıklama, clarification/tool-call sınırı, unavailable davranışı, çoklu çağrı sırası ve NFC/NFD.

30 kayıt kaliteyi istatistiksel olarak kanıtlamaz; yalnız pipeline ve erken hata canary'sidir. Her metrikte payda, hata-pozitif örnek sayısı, veri dilimi ve exact-binomial/Wilson güven aralığı raporlanır. `critical recall ≥99%` gibi otomasyon yetkisi ancak yeterli pozitif örnek ve önceden dondurulmuş güven aralığı kriteriyle verilir.

## 15. Prompt geliştirme sırası

Her adımda yalnız bir değişken değişir:

1. minimal baseline,
2. leaf-only extraction + host merge,
3. sentinel koruması,
4. fidelity contract,
5. ölçülmüş 2–4 karşıt örnek,
6. bağımsız risk router + ihtiyaç-temelli research,
7. mini adversarial verifier,
8. güçlü judge,
9. controlled repair.

Günlük geliştirme dev + canary üzerinde yapılır; frozen gold yalnız sürüm promotion'ında açılır.

## 16. Model ve maliyet politikası

Run config şunları exact-byte hash ile sabitler: endpoint, provider, model/snapshot, system fingerprint, reasoning/thinking, temperature/top_p, token limitleri, SDK sürümü, prompt/output schema/field policy/protector/research/judge/repair/eval sürümleri.

İlk 30/100/250/400:

```text
DeepSeek → deterministic → {kör GPT mini + kör GPT güçlü} → insan
```

400 sonrasında mini tüm uygun kayıtlarda kalabilir; güçlü judge riskli/research/disagreement/random calibration diliminde çalışabilir. Ancak insanın tek tek kabul etmediği kayıt yalnız silver olur. Kalibrasyon eşiği bozulursa güçlü judge ve insan kapsaması otomatik %100'e döner.

Episode başına translation, retry, research, retranslation, mini, strong, repair ve insan süresi ayrı ölçülür. Config sorgu/token/retry/repair ve parasal üst sınır taşır; limit aşımı sessiz düşürme değil human/quarantine durumudur.
