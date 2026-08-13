# 02 — Veri sözleşmesi ve aşamalar

Uygulama notu: A1–A6'nın çekirdeği fixture/property testleriyle uygulanmıştır. Ayrıca NVIDIA When2Call'ın test ve train split'leri ile Salesforce xLAM 60k train kaynağı revision-pinned snapshot/ingest/canonical/audit pilotlarından geçti; tüm satırlar canonical veya gerekçeli karantina olarak muhasebeleştirildi. When2Call training hedeflerinde yalnız source-explicit `<TOOLCALL>`, açık ek-bilgi istemi ve açık tool-unavailable davranışı kabul edilir. A7'nin explicit JSON Pointer kanıtlı deterministik Pass 1'i, A8'in exact/near-duplicate ve conflict audit altyapısı, A9'un split-leakage/selection freeze sözleşmesi fixture düzeyinde uygulanmıştır. Gerçek insan review ve S400 membership yoktur. A10–A13 için canlı komut yüzeyi vardır: tarihsel 3 episode / 20 leaf pre-review canary'nin yanında, doğrulanmış `translation-prompt-0.3.0` / Flash 6-worker ile 50 conflict-free adaylık bounded automation regression'ı 46 translation, 541 accepted leaf ve 40 satırlık pending-HF review paketi üretti. v0.4 aday prompt aynı cohortta daha düşük kabul sonucu verdiği için production'a alınmadı; canlı çeviri yalnız `promotion_status=validated` prompt bundle ile çalışır. Üretim release'i yapılmamıştır. Güncel kapsam için [06 — Uygulama durumu](06-implementation-status.md) belgesine bakın.

## 1. Satırın anlamı

Her canonical satır, tek veya çok turlu bir konuşma prefix'idir ve **tek bir hedef assistant mesajında** biter. Aynı kaynak diyalogda birden fazla assistant karar noktası varsa bağlı ama ayrı episode satırları üretilir.

Alan adımız `conversation` olarak kalır. Bir eğitim framework'ü `messages` bekliyorsa sürümlü trainer adapter çalışma anında dönüştürür; ikinci bir canonical veri sözleşmesi oluşturulmaz.

```json
{
  "schema_version": "0.1.0",
  "episode_id": "ep_...",
  "source_episode_fingerprint": "sha256:...",
  "variant_id": "sha256:...",
  "parent_variant_id": null,
  "conversation": [
    {
      "role": "user",
      "content": "Kullanıcı mesajı",
      "reasoning_content": null,
      "thinking": null,
      "tool_calls": null,
      "images": null,
      "name": null,
      "tool_call_id": null
    },
    {
      "role": "assistant",
      "content": null,
      "reasoning_content": null,
      "thinking": null,
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {"city": "Istanbul"}
          }
        }
      ],
      "images": null,
      "name": null,
      "tool_call_id": null
    }
  ],
  "tools": [
    {
      "tool_id": "tool_...",
      "documentation_hash": "sha256:...",
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets current weather.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name."}
          },
          "required": ["city"],
          "additionalProperties": false
        },
        "strict": null
      }
    }
  ],
  "provenance": {
    "sources": [
      {
        "dataset_namespace": "dataset_a",
        "snapshot_id": "snap_...",
        "source_occurrence_id": "occ_...",
        "source_sequence": 1,
        "source_native_id": null,
        "raw_record_sha256": "sha256:...",
        "observed_paths": ["/query", "/tools", "/tool_calls"]
      }
    ],
    "pipeline": {
      "version": "0.1.0",
      "run_event_id": "run_..."
    },
    "transformations": []
  },
  "annotations": {
    "source_conversation_id": "source-dialog-1",
    "target_message_index": 1,
    "parent_episode_id": null,
    "decision": {
      "action": "tool_call",
      "call_shape": "single",
      "call_ids": ["call_001"],
      "resolved_tool_ids": ["tool_..."],
      "missing_required_parameters": [],
      "evidence_status": "source_explicit"
    },
    "trajectory_state": "awaiting_tool",
    "execution_topology": "unknown"
  },
  "quality": {
    "state": "unreviewed",
    "flags": []
  }
}
```

`release_id` canonical satıra yazılmaz. Bir episode birden fazla release'te bulunabileceği için kullanıcı dostu sıra numarası release membership manifestinde tutulur.

## 2. Mesaj ve null sözleşmesi

Canonical message her zaman aynı anahtarları taşır; fakat kaynakta olmayan bir system mesajı sessizce eklenmez. Kaynak gerçekten boş system mesajı içeriyorsa `content=""` korunur.

Cardinality kuralları: `conversation` nonempty list, `tools` zorunlu listedir ve gerçekten tool sunulmayan episode'da `[]` olur; release/accepted satırda `provenance.sources` nonempty; `quality.flags` ve `provenance.transformations` zorunlu listelerdir.

Rol matrisi:

| Rol | `content` | `tool_calls` | `tool_call_id` | `name` |
|---|---|---|---|---|
| system/developer/user | string veya kaynakta yoksa null | null | null | null |
| assistant | content/reasoning/tool_calls alanlarından en az biri dolu | liste veya null | null | null |
| tool | string zorunlu | null | zorunlu | kaynakta varsa çağrı adıyla aynı |

Değer semantiği:

- `null`: canonical alanda değer yoktur.
- `[]`: kaynak açıkça boş liste sağlamıştır.
- `""`: kaynak açıkça boş metin sağlamıştır.
- “Kaynakta yok” ile “role uygulanamaz” ayrımı `provenance.sources[].observed_paths` ve source adapter field-map kaydıyla yapılır.
- Direct/final answer'da source açık `tool_calls=[]` sağladıysa boş liste korunur; sağlamadıysa `null` kalır.
- Kaynak sağlamıyorsa `reasoning_content` ve `thinking` daima `null`; biri diğerine kopyalanmaz ve sentetik reasoning üretilmez.

Görsel referansı varsa message-local liste şu typed yapıyı kullanır:

```text
ImageRef {id, uri, mime_type, sha256, width, height, anchor}
```

`uri` repo-relative veya içerik-adresli olmalı; base64 JSONL içine gömülmez. `anchor`, görselin metindeki sıra/konumunu korur.

## 3. Konuşma durum makinesi

`trajectory_state` değerleri:

- `complete`
- `awaiting_tool`
- `truncated`
- `failed`

Kurallar:

1. `assistant.tool_calls[*].id` episode içinde benzersizdir.
2. Her çağrının function adı o satırdaki `tools` listesinde tam bir kez çözülmelidir.
3. `role=tool` mesajı daha önce açılmış ve henüz cevaplanmamış `tool_call_id`ye bağlanır.
4. Orphan veya aynı call ID için ikinci tool sonucu geçersizdir.
5. Paralel sonuçların sırası serbesttir; ID eşleşmesi zorunludur.
6. Tool mesajında `name` varsa açılan çağrının function adıyla aynı olmalıdır.
7. `awaiting_tool`: hedef assistant tool call ile biter ve en az bir açık call vardır.
8. `complete`: final assistant mesajı vardır ve açık çağrı kalmamıştır; toolsuz direct/clarification/unavailable episode'larında da açık çağrı yoktur.
9. `truncated` ve `failed` yalnız hedef assistant mesajı kaynakta explicit olarak kesilmiş/hatalıysa atanır ve release eligibility ayrıca değerlendirilir.
10. Argument nesnesi ilgili `function.parameters` şemasına uymuyorsa episode `invalid_behavior` quarantine sebebi alır; clarification diye yeniden etiketlenmez.
11. Source trace tool sonucunda bitiyor fakat sonraki assistant hedefi yoksa lossless audit envelope'da korunur; v1 SFT canonical/release episode'u üretilmez.

### Kaynak kanıt yeterliliği matrisi

| Kaynakta bulunanlar | Canonical son | `trajectory_state` | Kullanım |
|---|---|---|---|
| user + assistant tool call | assistant `tool_calls` | `awaiting_tool` | tool selection ve argument prediction |
| user + assistant text | assistant `content` | `complete` | clarification/unavailable/direct answer |
| tool call + gerçek tool result, final yok | canonical training episode üretilmez | audit-only | source trace korunur, hedef uydurulmaz |
| tool call + gerçek result + source-backed final | final assistant | `complete` | grounded final-answer eğitimi |

Bir sonraki mesaj yalnız kaynak kanıtı varsa eklenir. Model tarafından sonradan üretilen result/final, `evidence_status=source_explicit` alamaz.

`quality.state` en az `unreviewed`, `deterministic_failed`, `model_review`, `human_accepted`, `human_rejected`, `quarantined` değerlerini taşır. Gold membership yalnız `human_accepted`; silver membership sürümlü policy ile model kapılarını geçen fakat insan tarafından tek tek kabul edilmemiş varyantları seçebilir.

## 4. Kimlik zinciri — ID önce

### 4.1 Snapshot

Tek dosya yerine çok dosyalı kaynak da desteklenir. Relative path'e göre sıralanmış file manifest JCS ile hash'lenir:

```json
{
  "id_version": 1,
  "dataset_namespace": "...",
  "source_revision": "...",
  "files": [{"relative_path": "...", "sha256": "...", "size": 0}]
}
```

`snapshot_id`, bu nesnenin SHA-256'sıdır. Manifest dataset/config/split, revision, lisans ve toplam satır sayısını da taşır.

### 4.2 Fiziksel occurrence ve sıra numarası

Her kayıt parse edilmeden önce byte sınırlarından kimlik alır:

```text
source_occurrence_id = sha256(JCS({
  id_version,
  snapshot_id,
  relative_file_path,
  byte_offset
}))
```

Aynı anda `source_sequence=1..N` verilir; bu kullanıcı dostu numara yalnız o snapshot içinde anlamlıdır. `raw_record_sha256`, satır sonu hariç tam fiziksel bytes'tan hesaplanır.

Strict parse sonrasında kaynak güvenilir native ID sağlıyorsa `source_native_id` ayrı alan olarak eklenir. Snapshotlar arası eşleştirme native ID, raw hash ve canonical fingerprint birlikte değerlendirilerek yapılır; line number kalıcı kimlik sayılmaz.

### 4.3 Episode ve varyant

```text
episode_id = sha256(JCS({
  id_version,
  source_occurrence_id,
  source_conversation_id,
  target_message_index,
  episode_schema_version
}))
```

`episode_id` İngilizce→Türkçe ve repair boyunca sabit kalır.

Cross-source exact duplicate için provenance ve source-specific ID'ler dışarıda bırakılır:

```text
source_episode_fingerprint = sha256(JCS({
  source_language_conversation,
  presented_tools_in_source_order,
  target_decision_and_output
}))
```

Aynı içeriğin farklı kaynak occurrences'ları bu fingerprint altında provenance alias'ı olur; ikinci eğitim satırı olarak seçilmez.

```text
variant_id = sha256(JCS({
  episode_id,
  conversation,
  tools,
  annotations_without_volatile_fields
}))
```

İngilizce, Türkçe ve repair çıktıları farklı `variant_id` taşır. Yeni varyant `parent_variant_id` ile bir önceki girdiye bağlanır. Membership `episode_id`, release logical hash ise seçilmiş `variant_id` üzerinden kurulur.

## 5. No-repeat, duplicate, conflict ve idempotency ayrımı

Bu kavramlar tek hash altında birleştirilmez.

### Candidate membership no-repeat

Yeni kaynak seçimi şu kimliklerle append-only JSONL olay geçmişine anti-join edilir:

- `source_occurrence_id`
- `source_native_id` aynı snapshot/config bağlamında
- `raw_record_sha256`
- `source_episode_fingerprint`

Rejected/quarantined occurrence otomatik olarak tekrar “yeni” sayılmaz. Yeni normalizer/politika ile tekrar işleme ancak explicit `supersedes_attempt_id` event'iyle açılır.

No-repeat görünümü veritabanında tutulmaz. Her çalıştırmada JSONL event shard'ları streaming okunur ve yalnız gerekli hash→owner kümeleri bellekte yeniden kurulur. Aynı giriş ve event geçmişi aynı indeks JSON artifact'ını üretir. İndeks silinebilir bir türevdir; olay JSONL'leri doğruluk kaynağıdır.

### Model-call idempotency

```text
attempt_key = sha256(JCS({
  episode_id,
  input_variant_id,
  stage,
  prompt_hash,
  evidence_hash,
  provider,
  resolved_model_snapshot,
  decode_config
}))
```

Aynı attempt key accepted ise provider tekrar çağrılmaz. Prompt/config değişikliği yeni kaynak seçimi değil, aynı episode için yeni attempt'tir.

### Karşılaştırma fingerprint'leri

- `structural_context_fingerprint`: konuşma context'i + sıra-bağımsız structural tool ID kümesi.
- `presented_context_fingerprint`: konuşma context'i + tool ID + documentation hash + modelin gerçekten gördüğü tool sırası.
- `ordered_behavior_fingerprint`: karar + sıralı calls/arguments + clarification bilgisi.
- `call_multiset_fingerprint`: aynı turdaki çağrıların sıra-bağımsız çokluk kümesi.

`source_episode_fingerprint`, presented context ve hedef behavior/output'un source-independent exact birleşimidir. Context veya behavior tek başına membership anti-join anahtarı değildir.

Hard behavior conflict yalnız `presented_context_fingerprint` aynı ve davranış gerçekten farklıysa verilir. Yalnız call sırası farklı, multiset aynı ve topology `unknown` ise `order_ambiguity_review` oluşur; otomatik quarantine/drop yapılmaz.

## 6. Tool schema normalizasyonu

- Object anahtarları recursive sıralanır.
- `required`, `type`, `enum`, `dependentRequired` gibi sıra-semantiksiz listeler canonical değere göre sıralanır.
- `prefixItems`, tuple schema ve diğer sıra-semantikli listeler korunur.
- `description`, `title`, `$comment`, `examples` structural kimliğe girmez; `documentation_hash`e girer.
- `default`, property adı, type, constraints ve validation semantics structural kimliğe girer.
- Remote `$ref` çalışma sırasında indirilmez; snapshot içindeki sabit bundle'a çözülür veya review/quarantine edilir.
- `allOf`, `anyOf`, `oneOf`, `$defs`, recursive `$ref`, duplicate enum ve vocabulary/keyword davranışı versioned keyword-policy tablosunda açıkça tanımlanır.
- Desteklenmeyen keyword sessizce atılmaz. Raw schema hash korunur ve kayıt review/quarantine edilir.
- Hem `raw_schema_hash`, `semantic_schema_hash`, `documentation_hash` hem `normalizer_version` saklanır.

### 6.1 Kararlı diagnostic sözleşmesi

Her doğrulama hatası yalnız serbest metinle değil, makinece işlenebilir ve sürümlü bir diagnostic kaydıyla raporlanır:

```json
{
  "schema_version": "diagnostic-0.1.0",
  "diagnostic_catalog_version": "0.1.0",
  "code": "SOURCE_ARG_NOT_GROUNDED",
  "stage": "source_semantic",
  "severity": "error",
  "source_occurrence_id": "occ_...",
  "episode_id": "ep_...",
  "json_pointer": "/conversation/1/tool_calls/0/function/arguments/city",
  "source_line": 42,
  "message": "Argument value has no permitted source evidence.",
  "retryable": false,
  "evidence_refs": []
}
```

Kod aileleri `PARSE_`, `SCHEMA_`, `TOOL_`, `SOURCE_`, `DEDUP_`, `CONFLICT_`, `TRANSLATION_`, `RESEARCH_`, `MEMORY_`, `QUALITY_`, `RENDER_` ve `RELEASE_` öneklerini kullanır. Mesaj metni geliştirilebilir; fakat bir kodun anlamı aynı `diagnostic_catalog_version` içinde değiştirilemez. Bilinmeyen diagnostic kodu fail-closed hatadır. CLI insan-okunur metin ve JSON çıktı sunar; doğrulama hatası exit code `1`, I/O veya config hatası exit code `2` üretir.

## 7. Kesin veri aşamaları

### A0 — Contract freeze

Schema, role/state machine, field policy, karar taksonomisi, tool keyword-policy, lisans/PII politikası ve kalite eşikleri onaylanır. Bunlar tamamlanmadan model çağrısı yapılmaz.

### A1 — Manifest ve event altyapısı

İlk kod fazında kurulur. İki ayrı kayıt vardır:

- deterministic content/membership manifest: input/output logical ve physical hash'ler, row accounting, schema/code/config/lock hash'leri; aynı girdide byte-identical.
- append-only run event: run ID, timestamp, duration, provider attempt ve parent manifest ID; doğal olarak byte-identical olmak zorunda değildir.

Publish `temp → validate → atomic final` ile yapılır. Var olan hedefe overwrite hatadır.

### A2 — Source register ve snapshot

Lisans, revision, config/split, file-root hash, dosya ve satır sayıları dondurulur. İzin verilmeyen lisans veya split release hattına giremez.

### A3 — ID-first bronze ingest

Önce physical `source_occurrence_id`, `source_sequence` ve raw hash; sonra strict JSON parse; sonra native ID çıkarımı yapılır. Duplicate key, NaN/Infinity, UTF-8, bozuk Unicode, eksik alan ve boyut ihlalleri ham kanıtla quarantine edilir.

### A4 — Source adapters

Her kaynak küçük ve testli adapter ile lossless raw envelope'a dönüştürülür. Adapter semantik karar vermez; field map ve observed paths üretir.

### A5 — Tool registry

Tool tanımları normalize edilir; tool ID, documentation hash, cross-source exact eşleşme ve schema conflict raporlanır.

### A6 — Canonical target episodes

Her source conversation'dan hedef assistant mesajlarında biten episode'lar çıkarılır. Parent episode bağlantısı, decision evidence ve trajectory state atanır.

### A7 — Kaynak semantik kanıtı, Pass 1

İngilizce canonical episode çeviriden önce kaynak davranışı açısından kontrol edilir:

- Seçilen tool kullanıcı isteğiyle uyumlu mu?
- Çağrı argument değerleri kullanıcı bağlamından veya açık source evidence'dan çıkarılabiliyor mu?
- Kaynakta olmayan değer uydurulmuş mu?
- Required/type/enum/schema kuralları geçerli mi?
- Clarification gerçekten eksik zorunlu veya ayırt edici bilgiyi mi soruyor?
- `tool_unavailable`, episode'da sunulan tool listesiyle tutarlı mı?
- Birden fazla makul tool/behavior varsa ambiguity kanıtı var mı?

Kapı iki geçişlidir:

1. **Pass 1 — tüm kayıtlar:** structure, schema, exposed-tool, argument type/required/enum, evidence pointer ve label-origin kontrolleri deterministik çalışır.
2. **Pass 2 — selection frontier:** A8'in ranked reserve queue'sundaki adaylar A9'da kör source judge ile kanıtlanır ve insan tarafından adjudicate edilir. 400 valid kayıt dolana kadar sıradaki aday alınır.

Kaynak-türü özel kontroller:

- xLAM: çağrılan tool presented listede tam bir kez çözülmeli; arguments schema'yı geçmeli; her değer user/source evidence, açık normalizasyon veya izinli sabit policy ile izlenebilmeli; tool relevance ve alternatif-tool belirsizliği incelenmeli.
- clarification: sorulan bilgi gerçekten eksik/ayırt edici olmalı; çıkarılan missing parameter şemadaki required/path ile kanıtlanamıyorsa review.
- tool unavailable: presented tools içinde makul bir çözüm bulunmadığı kanıtlanmalı; yalnız kaynak label'ına güvenilmez.
- direct/final answer: cevap source context veya gerçek tool result tarafından desteklenmeli; uydurma sonuç kabul edilmez.

Her expected argument için provenance sınıfı zorunludur:

| `origin` | Anlamı |
|---|---|
| `explicit_user` | Değer mevcut user mesajında açıkça bulunur. |
| `prior_turn` | Önceki konuşma turundan gelir. |
| `tool_result` | Önceki gerçek tool sonucundan gelir. |
| `system_context` | Kaynakta açık sistem bağlamıdır. |
| `deterministic_default` | Tool contractında açık ve izinli defaulttur. |
| `derived` | Sürümlü deterministik dönüşümle elde edilir. |
| `must_not_infer` | Eksikse tahmin edilmesi yasaktır. |
| `unknown` | Kaynak kanıtı bulunamamıştır; otomatik valid olamaz. |

`derived` için transformation ID, input pointerları ve sonuç kanıtı gerekir. Modelin bir değeri makul bulması provenance değildir.

Her karar insan tarafından okunabilir bir JSONL evidence kaydıdır:

```json
{
  "schema_version": "source-evidence-0.1.0",
  "episode_id": "ep_...",
  "deterministic_checks": [],
  "argument_provenance": [
    {
      "call_id": "call_001",
      "argument_pointer": "/city",
      "origin": "explicit_user",
      "evidence_pointers": ["/conversation/0/content"]
    }
  ],
  "acceptable_behaviors": [
    {
      "action": "tool_call",
      "tool_ids": ["tool_..."],
      "authority": "source_explicit"
    }
  ],
  "forbidden_behaviors": ["guess_missing_parameter"],
  "claims": [
    {
      "kind": "argument_grounding",
      "status": "supported",
      "source_pointers": ["/conversation/0/content"],
      "target_pointer": "/conversation/1/tool_calls/0/function/arguments/city"
    }
  ],
  "judge_verdict": "pass",
  "human_verdict": "source_valid",
  "review_event_id": "review_..."
}
```

Pass 1 sonucu `deterministic_pass`, `deterministic_fail` veya `needs_semantic_review` olur. Judge tool call'u değiştiremez, argument ekleyemez veya etiketi otomatik düzeltemez. Nihai `source_valid|source_review|source_invalid` A9'da atanır.

Birden fazla davranış yalnız kaynakta açıkça destekleniyorsa veya insan adjudication ile eklenmişse `acceptable_behaviors` içine girebilir. Judge/model tek başına alternatif gold yol üretemez. Aynı presented context'te davranış farkı kabul edilmiş alternatiflerle açıklanıyorsa conflict değildir; aksi halde conflict review devam eder.

### A8 — Exact duplicate, conflict ve near-duplicate adayları

| Presented context | Behavior | Eylem |
|---|---|---|
| aynı | aynı | duplicate alias; tek temsilci, tüm provenance korunur |
| aynı | farklı | behavior conflict review/quarantine |
| yakın | aynı | near-duplicate review adayı |
| yakın | farklı | yüksek öncelikli semantic conflict adayı |
| farklı | aynı tool | duplicate değildir |

Conflict kararları append-only adjudication event'idir: `keep_left`, `keep_right`, `keep_both_context_insufficient`, `source_error`, `policy_variant`, `defer`; reviewer/rubric sürümü ve superseded karar kaydedilir.

Önce character n-gram MinHash/LSH, ölçüm gerekirse multilingual embedding ile aday üretilir. Eşikler insan etiketli pair setiyle kalibre edilmeden otomatik drop yapılmaz.

Pass 1'i geçen ve unresolved hard conflict taşımayan episode'lar kaynak, action/call shape, tool ailesi, domain, uzunluk ve tool sayısına göre deterministik ranked reserve queue'ya alınır.

### A9 — Kaynak semantik Pass 2, selection ve split freeze

Duplicate/near-duplicate bağlı bileşenleri aynı split'te kalır. A8 queue sırasındaki adaylar kör source judge ve insan adjudication'dan geçirilir; valid olmayanların gerekçesi korunur ve sıradaki aday alınır. 400 insan-adjudicated `source_valid` episode oluşunca S400 master membership bir kez dondurulur:

```text
S30=rank 1..30, S100=1..100, S250=1..250, S400=1..400
```

Pilot sırasında yeniden seçim yapılmaz. Prompt değişirse membership değil attempt kapısı sıfırlanır.

### A10 — Pre-egress güvenlik ve terminoloji risk yönlendirmesi

Her external provider/search çağrısından önce secret, PII, local path ve izin politikası taranır. Gönderilecek segmentler minimize/redact edilir. Bağımsız terim-risk router araştırma ihtiyacını belirler.

### A11 — Segment çevirisi ve araştırma

Önce bağlamlı insan-kabul edilmiş tam-segment JSONL memory kontrol edilir. Lookup key exact source bytes, field/argument policy, context, tool/documentation, decision ve locale policy'den oluşur. Tek aktif hedef varsa reuse adayıdır; sıfırsa model çevirir; birden fazlaysa `memory_conflict` insan kuyruğuna gider. Fuzzy veya kelime-parçası reuse yoktur.

Her translation config önce memory-off dev/canary/frozen promotion eval'ini geçer. Yalnız promotion sonrasında production lane memory kullanabilir. Production raporu `model_generated` ve `memory_reused` kayıtları ayrı ölçer. Reuse yeni episode'da provenance event'i üretir, deterministik ve model/insan kapılarını atlamaz. Araştırma yalnız riskli terimler için ayrı evidence sidecar'ı oluşturur. Teknik yapı host tarafından birleştirilir.

### A12 — Kalite ve insan kabulü

Deterministik validator → kör mini structured verifier → mini non-pass + deterministik pass örneklemi için kör güçlü semantik judge → training render/loss-mask → insan kabulü. Örneklem dışı mini `pass` yeterlidir; escalation'da güçlü judge son kararı verir. Her attempt append-only event'tir.

İnsan incelemesinden önce sürümlü trainer adapter kayıtları hedef model chat template'ine render eder ve şunları doğrular:

- `conversation → messages` dönüşümü,
- tool tanımlarının doğru görünmesi,
- arguments'ın iki kez JSON-string olmaması,
- tool-call/result ID bağlantıları,
- hedef assistant loss mask'i,
- truncation'ın tool şemasını veya hedef label'ı kesmemesi,
- Türkçe Unicode ve chat-template prefix kararlılığı.

Her render config şu exact değerleri hash'ler: model ailesi, tokenizer adı/revision, chat-template exact bytes, tool serialization sürümü, BOS/EOS ve generation-prompt ayarı, maximum sequence length, truncation ve label policy.

Her model-specific training view JSONL satırı denetlenebilir kalır:

```json
{
  "schema_version": "training-render-0.1.0",
  "render_id": "render_...",
  "episode_id": "ep_...",
  "variant_id": "sha256:...",
  "render_config_id": "sha256:...",
  "rendered_text": "...",
  "token_count": 128,
  "supervised_token_ranges": [[96, 127]],
  "supervised_token_sha256": "sha256:...",
  "target_message_index": 1,
  "truncation_status": "none",
  "validation_status": "passed"
}
```

`supervised_token_ranges` yarı-açık `[start, end)` aralıklarıdır; örnekte 96–126 tokenları supervised olur. `supervised_token_sha256`, bu aralıklardaki sıralı token ID'lerinin canonical byte temsilinden hesaplanır.

Loss-mask v1 yalnız hedef son assistant mesajını supervise eder. Önceki bütün mesaj tokenları maskelidir. Tool-call hedefinde function name, arguments ve gerekli assistant control/EOS tokenları sürümlü label policy'ye göre birlikte supervise edilir. Target boşsa, target/tool schema kesiliyorsa, arguments iki kere string oluyorsa veya render prefix source prefix ile hizalanmıyorsa fail olur.

Render/loss-mask testi geçmeyen kayıt gold release'e giremez.

### A13 — Release

Frozen membership + accepted variant'lardan canonical JSONL üretilir. Hugging Face export etkinse aynı JSONL'den Parquet türetilir. Logical dataset hash, dosya hashleri, row count, schema/prompt/model/code sürümleri ve Dataset Card yayınlanır.

## 8. Split leakage politikası

Random satır split'i yapılmaz. Duplicate ve near-duplicate grafiğinin bağlı bileşeni tek bölüme atanır. Aynı presented context'in farklı davranış varyantları conflict çözülmeden hiçbir split'e girmez.

## 9. JSON/JSONL depolama ve isteğe bağlı Hugging Face export

Çalışma dosyaları:

```text
manifests/*.json
events/<stage>/<run_id>-*.jsonl
indexes/<index_name>-*.jsonl
bronze/<snapshot_id>-*.jsonl
canonical/<manifest_id>-*.jsonl
quarantine/<stage>/<run_id>-*.jsonl
reports/*.json
reviews/*.jsonl
memory/accepted_segments-*.jsonl
training_views/<render_config_id>-*.jsonl
```

Her JSONL satırı kendi schema sürümünü ve kimliğini taşır. Büyük dosyalar deterministik hash-prefix veya sabit kayıt sayısıyla shard edilir; shard listesi bir JSON manifestte sıralı tutulur. Tek bir ortak dosyaya eşzamanlı append yapılmaz: her run yeni geçici shard yazar, satır/hash kontrolünden sonra atomik olarak publish eder. CLI dosyaları streaming okur; tüm dataseti RAM'e alma zorunluluğu yoktur. JSON array şeklinde dev dosyalar kullanılmaz.

`indexes/*.jsonl` yalnız hızlandırıcı türevlerdir. Silinirse `events`, `canonical` ve manifestlerden birebir yeniden oluşturulabilir. `memory/accepted_segments-*.jsonl` yalnız insan-promoted tam segmentleri ve exact bağlam/policy hash'lerini taşır; kelime bazlı sözlük veya kör replace kaynağı değildir. `training_views` canonical truth değildir; model/template config'e göre tekrar üretilebilir.

Faz 1'de JSON Schema artifact'ları yazılır. Canonical JSONL, olay JSONL'leri, indeks ve manifestlerin her biri ayrı strict schema'ya sahip olur.

Parquet export açılırsa tam fiziksel Arrow schema ayrıca üretilir. `conversation` ve `tools` açık `list<struct<...>>`; değişken `arguments`, `parameters` ve `extra` Arrow JSON extension olur. Exact minimum PyArrow ve Hugging Face Datasets sürümleri `uv.lock` ile pinlenir.

Canonical JSONL kapısı her full logical row için strict schema, JCS hash, row count ve shard hash doğrular. Parquet export etkinse ek round-trip kapısı yalnız ID'lere bakmaz:

1. JSONL → explicit-schema Parquet,
2. Parquet JSON extension alanlarını parse ederek full logical row reconstruction,
3. null/empty dahil field-by-field deep equality ve full-row JCS hash eşitliği,
4. row count, Arrow schema metadata ve shard hash kontrolü.

Hugging Face varsayılan config yalnız Parquet dosyalarını Viewer'a açar. Canonical JSONL ayrı `raw` config/artifact altında tutulur; auto-discovery ile iki kez dataset olarak görünmez. Release build ZSTD, page index, ölçülmüş row-group ve shard hedeflerini manifestte kaydeder.

Hugging Face/Parquet hedeflenmiyorsa A13 doğrudan canonical JSONL + manifest ile tamamlanabilir. Parquet export hiçbir önceki aşamanın çalışması için zorunlu değildir.

Trainer adapter sürümlüdür ve model ailesine göre:

- `conversation → messages`,
- tool result ID bağlantıları,
- multimodal typed content/top-level images,
- chat template prefix kararlılığı

testlerini çalıştırır. Adaptasyon canonical veriyi yerinde değiştirmez.
