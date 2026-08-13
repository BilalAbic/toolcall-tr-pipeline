# 06 — Faz 1–5 ve sınırlı canlı provider yürütme durumu

Tarih: 2026-08-13

Bu belge, plan belgelerindeki hedeflerle repoda gerçekten çalışan yüzeyi birbirinden ayırır. “Uygulandı” ifadesi kod ve test kanıtı bulunan davranışları belirtir. Canlı kanıt sabit sentetik smoke'lar, 3 episode'luk sentetik çeviri batch'i ve 1 girdilik sentetik judge batch'iyle sınırlıdır. NVIDIA When2Call'ın test ve train split'leri ile Salesforce xLAM 60k train kaynağında revision-pinned modelsiz pilotlar tamamlanmıştır; insan adjudication, gerçek kaynak model egress'i veya release anlamına gelmez.

İki kaynağın tam revision/hash/pilot sayımları [07 — Gerçek kaynak pilot kayıtları](07-source-pilots.md) belgesindedir.

## Güvenlik sınırı

- `configs/pipeline.toml` içinde `providers.enabled=false` ve `network_egress_enabled=false` zorunludur.
- Config loader bu iki kapıdan biri açılırsa fail-closed hata verir.
- `secure_transport.py` yalnız public HTTPS, no-proxy/no-redirect, bounded response ve redacted hata yüzeyiyle çalışır. `credentials.py` yalnız allow-list'li provider anahtarını `.env` veya process environment'tan çözer; anahtar loglanmaz.
- `deepseek_adapter.py` yalnız `https://api.deepseek.com/chat/completions` ve V4 modellerini; `openai_judge.py` yalnız `https://api.openai.com/v1/responses` ile belirtilen judge modellerini kabul eder.
- `live_preflight.py` secret/PII/local-path/private endpoint bulgularını transport öncesi reddeder. Varsayılan pipeline config yine çevrimdışıdır.
- Varsayılan configte provider/eval kapalıdır. `translate` ve `evaluation run` yalnız explicit `--live`, non-default config, disjoint output root ve preflight ile çalışır; review/release yalnız haricî insan kararlarını doğrular.
- Testler `tests/fixtures/` altındaki küçük sentetik JSONL dosyalarını ve pytest geçici dizinlerini kullanır.
- Gerçek kaynak snapshot/pilot artifactı yalnız yerel ve ignore edilmiş artifact kökü altında üretildi; kaynak değiştirilmedi ve hiçbir gerçek kaynak satırı modele gönderilmedi. Gerçek insan kararı, S30/S400 seçimi veya release artifactı üretilmemiştir.

## Teslim matrisi

| Alan | Durum | Kod/artifact kanıtı | Doğrulanan sınır |
|---|---|---|---|
| Proje iskeleti ve kilitli ortam | Uygulandı | `pyproject.toml`, `uv.lock`, `src/toolcall_tr/`, `tests/` | Python 3.12, `uv`, pytest/Hypothesis, Ruff ve Pyright yapılandırması |
| Strict sözleşmeler | Uygulandı | `src/toolcall_tr/models.py`, `source.py`, `artifacts.py`, `events.py`, `diagnostics.py` | Pydantic strict/frozen, extra alan reddi, role/call/state doğrulaması |
| Draft 2020-12 şemaları | Uygulandı | `scripts/export_schemas.py`, `schemas/0.1.0/*.schema.json` | Seksen iki immutable versioned artifact; dialect ve meta-schema kontrolü |
| Diagnostic catalog | Uygulandı | `src/toolcall_tr/data/diagnostic_catalog.json` | Bilinen kodların anlamı catalogdan gelir; bilinmeyen kod fail-closed |
| Canonical JSON/hash/ID | Uygulandı | `src/toolcall_tr/hashing.py`, `ids.py` | RFC 8785/JCS, SHA-256, key-order invariance, non-finite sayı reddi |
| Artifact manifesti | Uygulandı | `src/toolcall_tr/artifacts.py`, `shards.py` | Content-addressed isim, dengeli row accounting, validate-before-publish, overwrite yasağı, idempotent resume |
| Event günlüğü | Uygulandı | `src/toolcall_tr/events.py` | Tek olaylık append-only shard, sequence/previous-hash zinciri, tamper detection |
| Source snapshot ve JSON-dizi dönüşümü | Uygulandı | `src/toolcall_tr/source.py`, `source_array.py` | JSONL file hash/size/row count, strict JSON-dizi → immutable JSONL, revision/lisans metadata ve mutation detection |
| Salesforce xLAM 60k kaynak pilotu | Uygulandı, gated erişimle | `src/toolcall_tr/adapters/xlam60k.py`, `pilot.py` | Revision `26d14e…7866`, train: 60.000 source-valid satırdan 57.718 canonical, 2.282 karantina ve 6 unresolved hard conflict; model/provider çağrısı yok |
| ID-first bronze ingest | Uygulandı | `src/toolcall_tr/source.py`, `jsonio.py` | Parse öncesi byte offset/sequence/occurrence ID; strict JSON ve quarantine |
| Source adapterları | Uygulandı | `src/toolcall_tr/adapters/base.py`, `no_tool.py`, `when2call.py`, `when2call_training.py`, `xlam.py`, `xlam60k.py` | `base.py` strict arayüz/yardımcılardır; beş concrete adapter xLAM fixture/60k, no-tool ve When2Call test+training source targetlarını kayıpsız eşler. Adapter kaynak davranışını tamir etmez veya icat etmez. |
| Tool registry | Uygulandı | `tool_registry.py` | Draft 2020-12 schema check, desteklenen-keyword allowlisti, remote-ref yasağı, structural/doc hash ayrımı |
| Canonicalizer/state machine | Uygulandı | `canonicalize.py`, `models.py` | Tool çözümleme ve argument schema check; source-backed tool-call-only kayıt `awaiting_tool`; sentetik result/final yok |
| Fingerprint/index yardımcıları | Temel uygulandı | `fingerprints.py`, `indexes.py` | Exact context/behavior kıyaslama, order-ambiguity routing, deterministik membership indexi |
| Faz 4 source evidence Pass 1 | Fixture düzeyinde uygulandı | `source_evidence.py`, `source evidence` CLI | Explicit JSON Pointer evidence; argument leaf coverage, `unknown`/`must_not_infer` fail-closed; judge çalışmaz, insan sonucu varsayılan `source_review` |
| Duplicate/conflict audit | Fixture düzeyinde uygulandı | `audit.py`, `adjudication.py`, `audit duplicates` CLI | Deterministik owner/alias, hard-conflict/order-ambiguity review, automatic drop yok; yalnız insan yetkili append-only adjudication sözleşmesi |
| Near duplicate ve split guard | Fixture düzeyinde uygulandı | `similarity.py`, `split_guard.py`, `audit near-duplicates` CLI | N-gram/Jaccard yalnız review adayı üretir; component farklı splitlerdeyse fail-closed |
| Selection freeze | Fixture düzeyinde uygulandı | `selection.py`, `phase4_config.py`, `select freeze` CLI | Human-adjudicated `source_valid`, grounding/conflict filtreleri, deterministic reserve queue ve strict S30/S100/S250/S400 prefixes; gerçek membership yok |
| Temel CLI | Uygulandı | `cli.py` | Register/validate/ingest/registry/canonicalize/source evidence/audit/select freeze/inspect/stats/events/diagnostics |
| Faz 5 field policy ve host merge | Uygulandı, gerçek canonical coverage ile | `field_policy.py`, `configs/field_policy.toml` | Tool/parameter açıklamaları `translate`; tüm adı geçmeyen argumentlar gözden geçirilmiş global `copy_exact` fallback ile modele kapalıdır. When2Call+xLAM 85.111 canonical episode üzerinde 849.064 segment / 0 unresolved policy error ağsız doğrulandı; technical fields immutable, coverage/merge fail-closed. |
| Faz 5 translation wire contract | Fixture düzeyinde uygulandı | `translation_contract.py` | Provider-shaped request/response yalnız local schema, sentinel order, NFC ve exact coverage için doğrulanır; istemci yok |
| Faz 5 pre-egress güvenlik | Fixture düzeyinde uygulandı | `egress_guard.py` | Secret/PII/local-path/private endpoint scan; offline configte clean payload dahi bloklu; HTTP/DNS/SDK yok |
| Faz 5 eval sözleşmesi | Fixture düzeyinde uygulandı | `eval_contract.py` | Atomic MQM finding + segment/path evidence, stdlib Wilson %95 coverage, deterministic rapor ve human-only gold eligibility |
| Faz 5 prompt/fake provider | Fixture düzeyinde uygulandı | `prompt_contract.py`, `configs/prompt_bundle.toml`, `fake_provider.py` | Content-addressed fixed-order prompt layers ve recorded strict JSON replay; HTTP/SDK/env secret yok |
| Faz 5 exact translation memory | Fixture düzeyinde uygulandı | `translation_memory.py` | Human-review event bağlı full segment entry; exact source/policy/context/tool/action lookup, fuzzy yok, active-target conflict fail-closed |
| Faz 5 terminology/research metadata | Fixture düzeyinde uygulandı | `research_policy.py` | Deterministic abbreviation/policy-risk router, public HTTPS/bütçe validator ve not-sent evidence sidecar; fetch yok |
| Canlı provider smoke | Uygulandı, sentetik sınırda | `secure_transport.py`, `credentials.py`, `deepseek_adapter.py`, `openai_judge.py`, `live_preflight.py`, `provider smoke` CLI | DeepSeek V4 çeviri ve OpenAI GPT-5.4-mini judge için gerçek endpoint smoke'u geçti; sabit sentetik içerik, explicit live config, endpoint allow-list, preflight ve local strict validation. Gerçek kaynak satırı bu endpointlere gönderilmedi. |
| Operasyonel kaynak pilotu | Uygulandı, gerçek kaynak kanıtıyla | `src/toolcall_tr/pilot.py`, `pilot run` CLI | When2Call revision `0582f…ace53`, tüm splitler: 27.952 satır / 27.393 canonical / 559 karantina / 136 cross-split conflict. Training için source-explicit `<TOOLCALL>`, açık clarification ve açık tool-unavailable hedefleri alınır. xLAM 60k revision `26d14e…7866`: 57.718 canonical / 2.282 karantina / 6 hard-conflict. Kaynak yeniden hash'lendi; model/provider çağrısı yok. |
| İnsan review ve Gold release | Uygulandı, gerçek karar bekliyor | `human_review_log.py`, `review submit-*`, `release build/validate` | Haricî tek-kayıt reviewer JSONL strict doğrulanır ve hash zincirine append edilir. Release her satır için local verdict, açık human acceptance ve linked review ID olmadan oluşmaz. |
| Provider attempt provenance | Uygulandı | `provider_provenance.py`, canlı adapterlar | Ham içerik veya credential saklamayan hash-only attempt kaydı; preflight/success/failure ve status sınıfları, otomatik retry bütçesi 0. |
| Operasyonel canlı çeviri | Uygulandı, gerçek-source policy coverage tamam; egress yetkisi kapalı | `src/toolcall_tr/operational_translation.py`, `translate` CLI | Yalnız field-policy tarafından izinli leaf'ler DeepSeek'e gider; gerçek canonical coverage ağsız geçti, source rehash ve host merge zorunlu. 3 sentetik no-tool episode'u gerçek endpointte geçti. İnsan review/selection kapıları nedeniyle gerçek kaynak egress'i, otomatik retry, Gold ve release yok. |
| Operasyonel canlı judge | Uygulandı, gerçek kaynak egress'i kapalı | `src/toolcall_tr/live_evaluation.py`, `evaluation run` CLI | Tam-leaf hash'li source/target input, role-specific OpenAI judge, immutable attempt/result/report artifactları. 1 sentetik girdi GPT-5.4-mini ile geçti; model çıktısı `gold_release_allowed=false` kalır. |
| Faz 6–7 render/loss-mask | Sözleşme kodu mevcut | `src/toolcall_tr/render_contract.py` | Enjekte edilmiş renderer/tokenizer, pinli config, teknik yapı ve final assistant payload eşleşmesi, tek target span ve truncation/offset uyuşmazlığında ret; model hub, chat-template/tokenizer indirme veya render CLI yok |
| Release manifest (Gold JSONL) | Fixture düzeyinde uygulandı | `release_contract.py`, `test_release_contract.py` | Sıralı yerel JSONL dosyalarının byte hash/row count doğrulaması, episode sırası ve açık human-review ID ile Gold üyeliği; gerçek release, Parquet, Dataset Card veya Hugging Face yok |
| Research/memory/repair ve gerçek dataset provider koşusu | Kısmen uygulandı | Faz 5+ | Prompt, provider adapter, live translation/judge batch sözleşmeleri vardır. Research fetch, automated repair, gerçek kaynak üzerinde çalıştırma ve batch auto-retry yok. |
| Pilot ve insan review çalıştırması | Kısmen uygulandı | Faz 6–8 | When2Call'ın tüm erişilebilir splitleri ve xLAM 60k üzerinde fail-closed modelsiz pilot tamamlandı. S30/S100/S250/S400 için yeterli ve yetkili insan review/adjudication yapılmadı; Gold kapalıdır. |
| Yayınlama/HF/Parquet | Uygulanmadı | Faz 9 | Release-manifest sözleşmesi vardır; Gold/silver artifact, Dataset Card, Parquet round-trip veya Hugging Face yayını yok |

## Schema artifactları

`uv run python scripts/export_schemas.py` şu immutable `0.1.0` sözleşmelerini üretir:

- `bronze-record.schema.json`
- `canonical-episode.schema.json`
- `canonical-quarantine.schema.json`
- `argument-path-policy.schema.json`
- `conflict-adjudication.schema.json`
- `conflict-candidate.schema.json`
- `content-manifest.schema.json`
- `diagnostic-catalog.schema.json`
- `diagnostic.schema.json`
- `egress-request.schema.json`
- `egress-violation.schema.json`
- `evaluation-report.schema.json`
- `evaluation-unit.schema.json`
- `exact-conflict-audit.schema.json`
- `exact-duplicate-group.schema.json`
- `field-policy.schema.json`
- `field-policy-segment.schema.json`
- `finding-count.schema.json`
- `gold-acceptance.schema.json`
- `human-evaluation-review.schema.json`
- `human-evaluation-review-entry.schema.json`
- `leaf-translation-record.schema.json`
- `live-evaluation-input.schema.json`
- `live-evaluation-result.schema.json`
- `live-evaluation-run.schema.json`
- `live-preflight-decision.schema.json`
- `model-evaluation-verdict.schema.json`
- `mqm-finding.schema.json`
- `outcome-summary.schema.json`
- `openai-judge-finding-output.schema.json`
- `openai-judge-output.schema.json`
- `operational-pilot.schema.json`
- `operational-pilot-tolerant.schema.json`
- `operational-translation.schema.json`
- `operational-translation-result.schema.json`
- `near-duplicate-candidate.schema.json`
- `phase4-config.schema.json`
- `prompt-bundle.schema.json`
- `prompt-layer.schema.json`
- `pre-egress-decision.schema.json`
- `protected-token.schema.json`
- `provider-attempt-record.schema.json`
- `research-budget.schema.json`
- `research-candidate.schema.json`
- `research-request.schema.json`
- `research-resolution.schema.json`
- `retry-budget-classification.schema.json`
- `release-dataset-file.schema.json`
- `release-gold-member.schema.json`
- `release-manifest.schema.json`
- `render-artifact.schema.json`
- `render-candidate.schema.json`
- `render-character-range.schema.json`
- `render-config.schema.json`
- `render-loss-mask.schema.json`
- `render-supervised.schema.json`
- `render-target-payload-range.schema.json`
- `render-token-range.schema.json`
- `run-event.schema.json`
- `selection-candidate.schema.json`
- `selection-manifest.schema.json`
- `segment-extraction.schema.json`
- `segment-memory-entry.schema.json`
- `segment-path-evidence.schema.json`
- `segment-translation.schema.json`
- `similarity-document.schema.json`
- `source-snapshot.schema.json`
- `json-array-conversion.schema.json`
- `source-evidence-input.schema.json`
- `source-evidence-request.schema.json`
- `source-evidence.schema.json`
- `split-leakage.schema.json`
- `translation-request.schema.json`
- `translation-response.schema.json`
- `translation-segment.schema.json`
- `translation-segment-result.schema.json`
- `memory-lookup-key.schema.json`
- `terminology-input.schema.json`
- `terminology-risk.schema.json`
- `tokenized-text.schema.json`
- `wilson-confidence-interval.schema.json`
- `coverage-summary.schema.json`

Her artifactın kök `$schema` değeri `https://json-schema.org/draft/2020-12/schema` olur. Export, şemanın JCS ile temsil edilebilir olduğunu kontrol eder; ayrıca doğrulama adımında her dosya `Draft202012Validator.check_schema` ile meta-schema kontrolünden geçirilir.

Pydantic modelleri sözleşmenin kaynak kodudur. `schemas/0.1.0/` elle düzenlenmez; model değişirse sürüm/migration kararı verilir ve export yeniden çalıştırılır.

## Test kapsamı ve bilinçli boşluklar

Mevcut testler şunları kapsar:

- strict JSONL okuma/yazma ve overwrite yasağı,
- duplicate JSON key, NaN/Infinity, invalid UTF-8/Unicode, boş/bozuk ve büyük kayıt quarantine,
- snapshot mutation, byte-offset occurrence ID ve deterministik sıra,
- strict model/schema round-trip ve extra alan reddi,
- xLAM tool-call-only ile no-tool karar fixtureları,
- tool schema normalization propertyleri,
- call order fingerprinti ve unknown-topology review davranışı,
- deterministic manifest, fixed-row shard, validation-before-publish ve event tamper detection,
- offline config, explicit-live translation/judge gate ve rebuildable membership indexi,
- explicit source-evidence pointer/origin kontrolleri ve CLI artifact yayını,
- exact alias/conflict ile human-only supersedes zinciri,
- non-destructive near-duplicate candidate retrieval ve connected-component split guard,
- 400 sentetik adayda human-gated S400 freeze, strict prefixler ve byte-identical rerun.
- field-policy extract/merge, strict segment wire contract ve pre-egress offline block.
- atomic MQM evidence, Wilson coverage report ve human-only gold acceptance.
- exact human-promoted segment memory ile not-sent terminology/research metadata.
- enjekte edilmiş transport ile Responses-biçimli strict request/response serialization, kapalı provider/egress gate ve malformed/uyuşmayan structured output reddi.
- allow-list'li secret resolver, public HTTPS/no-redirect transport, canlı preflight ve DeepSeek/OpenAI adapterlarının fixed-synthetic smoke yüzeyi.
- hash-only provider attempt kaydı, otomatik retry'nin kapalı olması ve sink hata durumunda fail-closed davranış.
- explicit kaynak JSONL için source-mutation/çıktı çakışması kapıları olan pilot kompozisyonu; human-review/release CLI'larında reviewer/verdict/Gold bağlantı denetimi.
- injected transport ile leaf-checkpoint/resume çevirisi ve full-leaf hash bağlı judge batch'i; preflight block, HTTP failure, no-auto-Gold ve output-boundary retleri.
- enjekte edilmiş renderer/tokenizer ile final assistant target-only loss mask, teknik yapı mutasyonu, truncation ve token-boundary uyuşmazlığında fail-closed ret.
- geçici yerel JSONL üzerinde release manifest round-trip, byte değişikliği, Gold insan kabul kimliği, episode sırası ve strict manifest kayıt reddi.

Bu testler gerçek dataset doğruluğunu, lisans uygunluğunu, gerçek source semantic kararı/insan adjudication'ı, çeviri kalitesini veya release hazırlığını kanıtlamaz.

## Tekrarlanabilir doğrulama

```powershell
uv sync --frozen
uv run python scripts/export_schemas.py
uv run pytest
uv run ruff check .
uv run pyright
```

Schema meta-doğrulaması:

```powershell
uv run python -c "import json; from pathlib import Path; from jsonschema import Draft202012Validator; files=sorted(Path('schemas/0.1.0').glob('*.schema.json')); [Draft202012Validator.check_schema(json.loads(p.read_text(encoding='utf-8'))) for p in files]; assert all(json.loads(p.read_text(encoding='utf-8'))['$schema']=='https://json-schema.org/draft/2020-12/schema' for p in files); print(f'{len(files)} Draft 2020-12 schema doğrulandı')"
```

## Sonraki uygulama sırası

1. When2Call'daki 559 ve xLAM 60k'daki 2.282 karantinayı; When2Call'daki 136 ve xLAM'daki 6 unresolved conflict'i kaynak kayıtlarını değiştirmeden yetkili insan review/adjudication ile ele al. Quarantine/conflict varsa karar olmadan devam etme.
2. Canonical survivor'lar üzerinde Faz 4 source evidence/selection artifactlarını üret; source-review/adjudication kararlarını yalnız yetkili insanlarla append-only loga al ve S400'ü ancak tüm kapılar kanıtlandıktan sonra dondur.
3. Render/loss-mask sözleşmesini gerçek hedef renderer/tokenizer ile ek kabul testlerinden geçir; target-only supervision sınırını kanıtla.
4. Küçük, onaylı pilot membership sağlandığında `translate` ve `evaluation run` komutlarını preflight ve hash-only attempt kayıtlarıyla çalıştır; model çıktısını Gold kabulü sayma.
5. Reviewer kararlarını `review submit-evaluation` ile zincire ekle. Her Gold satırını local model verdict + explicit human acceptance ile bağladıktan sonra `release build`/`release validate` çalıştır; Parquet/HF yayını için ayrıca karar ve kabul kapısı gerekir.

API anahtarı eklemek tek başına gerçek kaynak egress'i izni vermez. Varsayılan config çevrimdışıdır; canlı ağ erişimi explicit `--live`, non-default config, allow-list transport, preflight ve disjoint output ile mümkündür. Gerçek kaynak için insan-review/selection kapıları geçilmeden `translate` veya `evaluation run` çalıştırılmaz.
